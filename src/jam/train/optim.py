from dataclasses import dataclass

import torch


@dataclass
class OptimConfig:
    lr: float
    lr_agent: float | None = None
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    grad_clip: float = 1.0


def build_optimizer(model, cfg):
    world, agent = [], []
    for name, parameter in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "").replace("_orig_mod.", "")
        if parameter.requires_grad:
            (agent if name.startswith("agent.") else world).append(parameter)
    groups = []
    for name, parameters, lr in [
        ("world", world, cfg.lr),
        ("agent", agent, cfg.lr_agent or cfg.lr),
    ]:
        if parameters:
            groups.append({"params": parameters, "lr": lr, "name": name})
    return torch.optim.AdamW(groups, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay)
