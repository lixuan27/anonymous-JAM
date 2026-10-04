from __future__ import annotations

from dataclasses import dataclass

import torch

from .flow import FlowSchedule, JointNoisePlane, masked_mse
from .jam import JointActionWorldModel


@dataclass
class ObjectiveConfig:
    lambda_world: float = 1.0
    lambda_action: float = 1.0
    lambda_consequence: float = 1.0
    world_shift: float = 5.0
    action_shift: float = 5.0
    weighting: str = "bell"


def effective_levels(s: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    return torch.where(present.to(torch.bool), s, torch.ones_like(s))


class JointFlowObjective:
    def __init__(self, cfg: ObjectiveConfig, clean_frames: int = 1):
        self.cfg = cfg
        self.world_flow = FlowSchedule(shift=cfg.world_shift, weighting=cfg.weighting)
        self.action_flow = FlowSchedule(shift=cfg.action_shift, weighting=cfg.weighting)
        self.plane = JointNoisePlane(self.world_flow, self.action_flow)
        self.clean_frames = clean_frames

    def __call__(
        self, model: JointActionWorldModel, batch: dict, generator: torch.Generator | None = None
    ) -> dict:
        z0 = batch["latents"]
        a0 = batch["actions"]
        B = z0.shape[0]
        dev = z0.device
        draw = self.plane.draw(B, device=dev, generator=generator)
        s_w, s_a, s_c = (draw["s_world"], draw["s_action"], draw["s_consequence"])
        eps_w = torch.randn(z0.shape, device=dev, dtype=torch.float32, generator=generator)
        z_s = FlowSchedule.noise(z0.float(), eps_w, s_w.view(B, 1, 1, 1, 1))
        z_s[:, :, : self.clean_frames] = z0[:, :, : self.clean_frames].float()
        has_action = (
            batch["action_mask"].flatten(1).any(1).float()
            if "action_mask" in batch
            else torch.ones(B, device=dev)
        )
        s_a = effective_levels(s_a, has_action)
        eps_a = torch.randn(a0.shape, device=dev, dtype=torch.float32, generator=generator)
        a_s = FlowSchedule.noise(a0.float(), eps_a, s_a.view(B, 1, 1))
        c0 = batch.get("consequence")
        c_s = None
        if c0 is not None:
            has_c = (
                batch["consequence_mask"].flatten(1).any(1).float()
                if "consequence_mask" in batch
                else torch.ones(B, device=dev)
            )
            s_c = effective_levels(s_c, has_c)
            eps_c = torch.randn(c0.shape, device=dev, dtype=torch.float32, generator=generator)
            c_s = FlowSchedule.noise(c0.float(), eps_c, s_c.view(B, 1, 1))
        out = model(
            latents=z_s.to(z0.dtype),
            actions=a_s.to(a0.dtype),
            s_world=s_w,
            s_action=s_a,
            text=batch["text"],
            text_mask=batch.get("text_mask"),
            state=batch.get("state"),
            state_mask=batch.get("state_mask"),
            consequence=None if c_s is None else c_s.to(a0.dtype),
            s_consequence=s_c,
        )
        tgt_w = FlowSchedule.target(z0.float(), eps_w)
        per_frame = masked_mse(out.v_world, tgt_w, None, reduce_dims=(1, 3, 4))
        frame_valid = batch.get("latent_mask")
        if frame_valid is None:
            frame_valid = torch.ones(per_frame.shape, dtype=torch.bool, device=dev)
        frame_valid = frame_valid.clone()
        frame_valid[:, : self.clean_frames] = False
        w_per_sample = (per_frame * frame_valid).sum(1) / frame_valid.sum(1).clamp_min(1)
        loss_w = (w_per_sample * draw["w_world"]).mean()
        tgt_a = FlowSchedule.target(a0.float(), eps_a)
        a_per_sample = masked_mse(out.v_action, tgt_a, batch.get("action_mask"), reduce_dims=(1, 2))
        has_action = (
            batch["action_mask"].flatten(1).any(1).float()
            if "action_mask" in batch
            else torch.ones(B, device=dev)
        )
        loss_a = (a_per_sample * draw["w_action"] * has_action).sum() / has_action.sum().clamp_min(
            1.0
        )
        total = self.cfg.lambda_world * loss_w + self.cfg.lambda_action * loss_a
        logs = {"loss/world": loss_w.detach(), "loss/action": loss_a.detach()}
        if c0 is not None and out.v_consequence is not None:
            tgt_c = FlowSchedule.target(c0.float(), eps_c)
            c_per_sample = masked_mse(
                out.v_consequence, tgt_c, batch.get("consequence_mask"), reduce_dims=(1, 2)
            )
            has_c = (
                batch["consequence_mask"].flatten(1).any(1).float()
                if "consequence_mask" in batch
                else torch.ones(B, device=dev)
            )
            loss_c = (c_per_sample * draw["w_consequence"] * has_c).sum() / has_c.sum().clamp_min(
                1.0
            )
            total = total + self.cfg.lambda_consequence * loss_c
            logs["loss/consequence"] = loss_c.detach()
        logs["loss/total"] = total.detach()
        return {"loss": total, "logs": logs, "s_world": s_w, "s_action": s_a}
