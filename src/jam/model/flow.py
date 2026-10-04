from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Weighting = Literal["uniform", "bell"]


@dataclass
class FlowSchedule:
    shift: float = 5.0
    weighting: Weighting = "bell"
    bell_center: float = 0.5
    bell_sharpness: float = 2.0
    _weight_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.shift <= 0:
            raise ValueError("shift must be > 0")
        u = (torch.arange(20000, dtype=torch.float64) + 0.5) / 20000
        s = self.warp(u)
        w = self._raw_weight(s)
        self._weight_norm = float(w.mean())

    def warp(self, u: torch.Tensor) -> torch.Tensor:
        return self.shift * u / (1.0 + (self.shift - 1.0) * u)

    def unwarp(self, s: torch.Tensor) -> torch.Tensor:
        return s / (self.shift - (self.shift - 1.0) * s)

    def sample(
        self, n: int, *, device=None, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        u = torch.rand(n, device=device, generator=generator, dtype=torch.float32)
        return self.warp(u)

    def _raw_weight(self, s: torch.Tensor) -> torch.Tensor:
        if self.weighting == "uniform":
            return torch.ones_like(s)
        w = torch.exp(-self.bell_sharpness * (s - self.bell_center) ** 2)
        wmin = torch.exp(
            -self.bell_sharpness
            * torch.tensor(max(self.bell_center, 1 - self.bell_center), dtype=s.dtype) ** 2
        )
        return w - wmin

    def weight(self, s: torch.Tensor) -> torch.Tensor:
        return self._raw_weight(s.to(torch.float32)) / self._weight_norm

    @staticmethod
    def noise(x0: torch.Tensor, eps: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (1.0 - s) * x0 + s * eps

    @staticmethod
    def target(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return eps - x0

    def inference_levels(self, n_steps: int, *, start: float = 1.0) -> torch.Tensor:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        u = torch.linspace(start, 0.0, n_steps + 1, dtype=torch.float64)
        s = self.warp(u)
        s[-1] = 0.0
        return s.to(torch.float32)

    @staticmethod
    def euler(x: torch.Tensor, v: torch.Tensor, s: float, s_next: float) -> torch.Tensor:
        return x + v * (s_next - s)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    reduce_dims: tuple[int, ...],
) -> torch.Tensor:
    err = (pred.float() - target.float()) ** 2
    if mask is None:
        return err.mean(dim=reduce_dims)
    m = mask.to(err.dtype).expand_as(err)
    num = (err * m).sum(dim=reduce_dims)
    den = m.sum(dim=reduce_dims).clamp_min(1.0)
    return num / den


@dataclass
class JointNoisePlane:
    world: FlowSchedule
    action: FlowSchedule

    def draw(self, n: int, *, device=None, generator=None):
        sw = self.world.sample(n, device=device, generator=generator)
        sa = self.action.sample(n, device=device, generator=generator)
        sc = self.action.sample(n, device=device, generator=generator)
        return {
            "s_world": sw,
            "s_action": sa,
            "s_consequence": sc,
            "w_world": self.world.weight(sw),
            "w_action": self.action.weight(sa),
            "w_consequence": self.action.weight(sc),
        }
