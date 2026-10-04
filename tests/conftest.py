import os

import pytest
import torch

from jam.model.action_stream import AgentStream, AgentStreamConfig
from jam.model.backbone import read_backbone_config, resolve_backbone_type
from jam.model.jam import JointActionWorldModel
from jam.model.world_stream import LayerHub, WorldStream

TINY = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=2,
    attention_head_dim=16,
    in_channels=4,
    out_channels=4,
    text_dim=8,
    freq_dim=16,
    ffn_dim=64,
    num_layers=2,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=128,
)


@pytest.fixture(scope="session")
def backbone_type():
    path = os.environ.get("JAM_BACKBONE_CONFIG")
    if not path:
        pytest.fail(
            "Set JAM_BACKBONE_CONFIG to the local video-backbone config.json; weights are not required."
        )
    return resolve_backbone_type(read_backbone_config(path))


@pytest.fixture
def model(backbone_type):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    world = WorldStream(backbone_type(**TINY), LayerHub())
    agent = AgentStream(
        AgentStreamConfig(
            action_dim=6,
            state_dim=6,
            chunk=4,
            width=32,
            layers=2,
            heads=2,
            head_dim=16,
            ctx_dim=8,
            consequence_dim=6,
            consequence_steps=2,
            zero_init_heads=False,
        )
    )
    return JointActionWorldModel(world, agent)


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(1)
    shapes = {
        "latents": (2, 4, 3, 4, 4),
        "actions": (2, 4, 6),
        "state": (2, 6),
        "text": (2, 3, 8),
        "consequence": (2, 2, 6),
    }
    result = {key: torch.randn(shape, generator=generator) for key, shape in shapes.items()}
    for key in ("actions", "state", "consequence"):
        result[{"actions": "action_mask"}.get(key, key + "_mask")] = torch.ones_like(
            result[key], dtype=torch.bool
        )
    result["latent_mask"] = torch.ones(2, 3, dtype=torch.bool)
    result["text_mask"] = torch.ones(2, 3, dtype=torch.bool)
    return result
