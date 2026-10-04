import copy

import pytest
import torch

from jam.model.coupling import TokenBook, build_joint_mask
from jam.model.flow import FlowSchedule, JointNoisePlane, masked_mse
from jam.model.jam import pair_layers
from jam.model.objective import JointFlowObjective, ObjectiveConfig
from jam.train.checkpoint import load_checkpoint, save_checkpoint
from jam.train.loop import train_epoch
from jam.train.optim import OptimConfig, build_optimizer


def forward(model, batch):
    return model(
        batch["latents"],
        batch["actions"],
        torch.full((2,), 0.4),
        torch.full((2,), 0.7),
        batch["text"],
        batch["text_mask"],
        batch["state"],
        batch["state_mask"],
        batch["consequence"],
        torch.full((2,), 0.5),
    )


def test_mask():
    book = TokenBook(6, 2, 1, 4, 2)
    mask = build_joint_mask(book)
    assert mask[book.clean, book.clean].all()
    assert not mask[book.clean, 2:].any()
    assert mask[2:].all()


def test_layer_mapping():
    assert pair_layers(4, 2) == {0: 0, 2: 1}
    with pytest.raises(ValueError):
        pair_layers(2, 3)


def test_reciprocal_gradients(model, batch):
    output = forward(model, batch)
    assert output.v_world.shape == batch["latents"].shape
    assert output.v_action.shape == batch["actions"].shape
    assert output.v_consequence.shape == batch["consequence"].shape
    output.v_action.square().mean().backward()
    assert model.world.net.blocks[0].attn1.to_k.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    forward(model, batch).v_world.square().mean().backward()
    assert model.agent.blocks[0].k.weight.grad.abs().sum() > 0


def test_clean_observation_is_isolated(model, batch):
    reference = forward(model, batch)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["actions"] += 2
    changed["consequence"] -= 1
    changed["latents"][:, :, 1:] += 3
    result = forward(model, changed)
    torch.testing.assert_close(reference.v_world[:, :, :1], result.v_world[:, :, :1])
    assert not torch.allclose(reference.v_action, result.v_action)


def test_text_padding(model, batch):
    batch["text_mask"][:, -1] = False
    reference = forward(model, batch)
    batch["text"][:, -1] = 1000
    result = forward(model, batch)
    for name in ("v_world", "v_action", "v_consequence"):
        torch.testing.assert_close(getattr(reference, name), getattr(result, name))


def test_checkpoint_recomputation(model, batch):
    other = copy.deepcopy(model)
    other.gradient_checkpointing = True
    objective = JointFlowObjective(ObjectiveConfig())
    for net in (model, other):
        objective(net, batch, generator=torch.Generator().manual_seed(9))["loss"].backward()
    for a, b in zip(model.parameters(), other.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)


def test_missing_modalities(model, batch):
    batch["action_mask"][0] = False
    batch["consequence_mask"][1] = False
    objective = JointFlowObjective(ObjectiveConfig())
    before = objective(model, batch, generator=torch.Generator().manual_seed(8))
    batch["actions"][0] = 7
    batch["consequence"][1] = -5
    after = objective(model, batch, generator=torch.Generator().manual_seed(8))
    assert before["s_action"][0] == 1
    torch.testing.assert_close(before["loss"], after["loss"])
    after["loss"].backward()
    assert model.agent.consequence_out.weight.grad.abs().sum() > 0


def test_masked_coordinates():
    pred = torch.tensor([[[2.0, 999.0]]], requires_grad=True)
    mask = torch.tensor([[[True, False]]])
    loss = masked_mse(pred, torch.zeros_like(pred), mask, (1, 2)).sum()
    assert loss == 4
    loss.backward()
    assert pred.grad[0, 0, 1] == 0
    assert masked_mse(pred, pred, torch.zeros_like(mask), (1, 2)).item() == 0


def test_flow():
    flow = FlowSchedule()
    draw = JointNoisePlane(flow, flow).draw(128, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(draw["s_world"], draw["s_action"])
    assert not torch.equal(draw["s_action"], draw["s_consequence"])
    levels = flow.inference_levels(4)
    assert levels[0] == 1 and levels[-1] == 0
    assert (levels[1:] < levels[:-1]).all()
    clean, noise = torch.tensor([2.0]), torch.tensor([7.0])
    torch.testing.assert_close(flow.euler(noise, flow.target(clean, noise), 1, 0), clean)


def test_accumulation_and_partial_group(model, batch):
    other = copy.deepcopy(model)
    objective = JointFlowObjective(ObjectiveConfig())
    config = OptimConfig(lr=1e-3)
    optimizer, reference_optimizer = build_optimizer(model, config), build_optimizer(other, config)
    generator = torch.Generator().manual_seed(4)
    train_epoch(
        model,
        [batch, batch, batch],
        objective,
        optimizer,
        device=torch.device("cpu"),
        grad_accum=2,
        generator=generator,
    )
    generator.manual_seed(4)
    for group in ([batch, batch], [batch]):
        reference_optimizer.zero_grad(set_to_none=True)
        for item in group:
            (objective(other, item, generator=generator)["loss"] / len(group)).backward()
        torch.nn.utils.clip_grad_norm_(other.parameters(), 1.0)
        reference_optimizer.step()
    for a, b in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(a, b)


def test_training_and_resume(model, batch, tmp_path):
    objective = JointFlowObjective(ObjectiveConfig())
    optimizer = build_optimizer(model, OptimConfig(lr=1e-3))
    train_epoch(model, [batch], objective, optimizer, device=torch.device("cpu"))
    save_checkpoint(tmp_path / "checkpoint", model, optimizer, completed_epochs=1)
    expected_rng = torch.get_rng_state()
    train_epoch(model, [batch], objective, optimizer, device=torch.device("cpu"))
    expected = copy.deepcopy(model.state_dict())
    assert load_checkpoint(tmp_path / "checkpoint", model, optimizer) == 1
    assert torch.equal(torch.get_rng_state(), expected_rng)
    train_epoch(model, [batch], objective, optimizer, device=torch.device("cpu"))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key])


def test_invalid_text_mask(model, batch):
    batch["text_mask"][0] = False
    with pytest.raises(ValueError, match="valid text token"):
        forward(model, batch)


def test_batch_contract(model, batch):
    from jam.train.batch import validate_batch

    validate_batch(batch, model.agent.cfg)
    batch["action_mask"] = batch["action_mask"].float()
    with pytest.raises(ValueError, match="action_mask must be boolean"):
        validate_batch(batch, model.agent.cfg)


def test_cli(model, batch, tmp_path):
    import json
    import os
    import subprocess
    import sys
    from dataclasses import asdict
    from pathlib import Path

    from safetensors.torch import save_file

    model.world.net.save_pretrained(tmp_path / "backbone" / "transformer")
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"agent": asdict(model.agent.cfg)}))
    batches = tmp_path / "batches"
    batches.mkdir()
    save_file(batch, batches / "batch.safetensors")
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jam.train",
            "--backbone",
            str(tmp_path / "backbone"),
            "--config",
            str(config),
            "--batches",
            str(batches),
            "--epochs",
            "1",
            "--lr",
            "0.001",
            "--output",
            str(tmp_path / "output"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "loss/total" in result.stdout
    assert (tmp_path / "output/epoch-1/COMPLETE").is_file()
