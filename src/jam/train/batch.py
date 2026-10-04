import torch


def validate_batch(batch, agent_config):
    shapes = {key: tuple(value.shape) for key, value in batch.items()}
    if "latents" not in shapes or len(shapes["latents"]) != 5:
        raise ValueError("latents must have shape (B, C, T, H, W)")
    size, _, frames, _, _ = shapes["latents"]
    if size < 1 or frames < 2:
        raise ValueError("a batch needs samples and future latent frames")
    expected = {"actions": (size, agent_config.chunk, agent_config.action_dim)}
    if "state" in batch:
        expected["state"] = (size, agent_config.state_dim)
    if agent_config.consequence_dim:
        horizon = agent_config.consequence_steps or agent_config.chunk
        expected["consequence"] = (size, horizon, agent_config.consequence_dim)
    elif "consequence" in batch:
        raise ValueError("Motion Track input requires a configured Motion Track head")
    for key, shape in expected.items():
        if shapes.get(key) != shape:
            raise ValueError(f"{key} must have shape {shape}")
    if (
        len(shapes.get("text", ())) != 3
        or shapes["text"][0] != size
        or shapes["text"][1] < 1
        or shapes["text"][2] != agent_config.ctx_dim
    ):
        raise ValueError("text must have shape (B, L, text_dim)")
    masks = {
        "latent_mask": (size, frames),
        "text_mask": shapes["text"][:2],
        "action_mask": shapes["actions"],
    }
    masks.update({key + "_mask": shapes[key] for key in ("state", "consequence") if key in batch})
    for key, shape in masks.items():
        if shapes.get(key) != shape or batch[key].dtype != torch.bool:
            raise ValueError(f"{key} must be boolean with shape {shape}")
    if not batch["latent_mask"][:, 1:].any(dim=1).all():
        raise ValueError("each sample needs a valid future frame")
    if not batch["latent_mask"][:, 0].all():
        raise ValueError("the observed prefix must be valid")
    for key, value in batch.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values")
    return batch
