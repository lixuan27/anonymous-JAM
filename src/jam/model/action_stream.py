from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float()[..., None] * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class NoiseLevelEmbedding(nn.Module):
    def __init__(self, width: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.mlp(
            sinusoidal_embedding(s * 1000.0, self.freq_dim).to(self.mlp[0].weight.dtype)
        )


def rope_1d(
    positions: torch.Tensor, head_dim: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv = 1.0 / theta ** (torch.arange(half, dtype=torch.float64, device=positions.device) / half)
    ang = positions.to(torch.float64)[:, None] * inv[None, :]
    cos = torch.cos(ang).repeat_interleave(2, dim=-1)
    sin = torch.sin(ang).repeat_interleave(2, dim=-1)
    return (cos.float(), sin.float())


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = (x[..., 0::2], x[..., 1::2])
    c, s = (cos[None, :, None, 0::2], sin[None, :, None, 1::2])
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def layer_norm_fp32(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    w = norm.weight.float() if norm.weight is not None else None
    b = norm.bias.float() if norm.bias is not None else None
    return F.layer_norm(x.float(), norm.normalized_shape, w, b, norm.eps).type_as(x)


class CrossAttention(nn.Module):
    def __init__(self, width: int, ctx_dim: int, heads: int, head_dim: int):
        super().__init__()
        inner = heads * head_dim
        self.heads, self.head_dim = (heads, head_dim)
        self.q = nn.Linear(width, inner)
        self.k = nn.Linear(ctx_dim, inner)
        self.v = nn.Linear(ctx_dim, inner)
        self.o = nn.Linear(inner, width)
        self.norm_q = nn.RMSNorm(inner, eps=1e-06)
        self.norm_k = nn.RMSNorm(inner, eps=1e-06)

    def forward(
        self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor | None
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.norm_q(self.q(x)).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = (
            self.norm_k(self.k(ctx))
            .view(B, ctx.shape[1], self.heads, self.head_dim)
            .transpose(1, 2)
        )
        v = self.v(ctx).view(B, ctx.shape[1], self.heads, self.head_dim).transpose(1, 2)
        mask = None
        if ctx_mask is not None:
            mask = ctx_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class AgentBlock(nn.Module):
    def __init__(
        self,
        width: int,
        ctx_dim: int,
        heads: int,
        head_dim: int,
        ffn_mult: int = 4,
        eps: float = 1e-06,
    ):
        super().__init__()
        inner = heads * head_dim
        self.heads, self.head_dim = (heads, head_dim)
        self.norm_attn = nn.LayerNorm(width, eps=eps, elementwise_affine=False)
        self.q = nn.Linear(width, inner)
        self.k = nn.Linear(width, inner)
        self.v = nn.Linear(width, inner)
        self.norm_q = nn.RMSNorm(inner, eps=eps)
        self.norm_k = nn.RMSNorm(inner, eps=eps)
        self.o = nn.Linear(inner, width)
        self.norm_ctx = nn.LayerNorm(width, eps=eps, elementwise_affine=True)
        self.cross = CrossAttention(width, ctx_dim, heads, head_dim)
        self.norm_ffn = nn.LayerNorm(width, eps=eps, elementwise_affine=False)
        self.ffn = FeedForward(width, ffn_mult * width)
        self.ada = nn.Parameter(torch.randn(1, 6, width) / math.sqrt(width))

    def modulation(self, temb: torch.Tensor):
        return [t.squeeze(2) for t in (self.ada[:, None] + temb).chunk(6, dim=2)]

    def project_qkv(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        B, L, _ = x.shape
        h = (layer_norm_fp32(self.norm_attn, x).float() * (1 + scale) + shift).type_as(x)
        q = self.norm_q(self.q(h)).view(B, L, self.heads, self.head_dim)
        k = self.norm_k(self.k(h)).view(B, L, self.heads, self.head_dim)
        v = self.v(h).view(B, L, self.heads, self.head_dim)
        q = apply_rope(q, cos, sin).type_as(v)
        k = apply_rope(k, cos, sin).type_as(v)
        return (q, k, v)

    def finish(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        gate: torch.Tensor,
        c_shift: torch.Tensor,
        c_scale: torch.Tensor,
        c_gate: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.o(attn_out) * gate
        x = x + self.cross(layer_norm_fp32(self.norm_ctx, x), ctx, ctx_mask)
        h = (layer_norm_fp32(self.norm_ffn, x).float() * (1 + c_scale) + c_shift).type_as(x)
        return x + self.ffn(h) * c_gate


@dataclass
class AgentStreamConfig:
    action_dim: int = 80
    state_dim: int = 80
    chunk: int = 32
    width: int = 1024
    layers: int = 30
    heads: int = 24
    head_dim: int = 128
    ffn_mult: int = 4
    ctx_dim: int = 4096
    consequence_dim: int = 0
    consequence_steps: int = 0
    zero_init_heads: bool = True


class AgentStream(nn.Module):
    def __init__(self, cfg: AgentStreamConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.action_in = nn.Linear(cfg.action_dim, w)
        self.state_in = nn.Linear(cfg.state_dim, w)
        self.null_state = nn.Parameter(torch.zeros(1, 1, w))
        self.type_embed = nn.Embedding(3, w)
        self.time = NoiseLevelEmbedding(w)
        self.time_to_ada = nn.Sequential(nn.SiLU(), nn.Linear(w, 6 * w))
        self.ctx_in = nn.Sequential(
            nn.Linear(cfg.ctx_dim, w), nn.GELU(approximate="tanh"), nn.Linear(w, w)
        )
        self.blocks = nn.ModuleList(
            [AgentBlock(w, w, cfg.heads, cfg.head_dim, cfg.ffn_mult) for _ in range(cfg.layers)]
        )
        self.norm_out = nn.LayerNorm(w, eps=1e-06, elementwise_affine=False)
        self.ada_out = nn.Parameter(torch.randn(1, 2, w) / math.sqrt(w))
        self.action_out = nn.Linear(w, cfg.action_dim)
        if cfg.consequence_dim > 0:
            self.consequence_in = nn.Linear(cfg.consequence_dim, w)
            self.consequence_out = nn.Linear(w, cfg.consequence_dim)
        if cfg.zero_init_heads:
            nn.init.zeros_(self.action_out.weight)
            nn.init.zeros_(self.action_out.bias)
            if cfg.consequence_dim > 0:
                nn.init.zeros_(self.consequence_out.weight)
                nn.init.zeros_(self.consequence_out.bias)

    @property
    def consequence_steps(self) -> int:
        if self.cfg.consequence_dim <= 0:
            return 0
        return self.cfg.consequence_steps or self.cfg.chunk

    @property
    def n_tokens(self) -> int:
        return 1 + self.cfg.chunk + self.consequence_steps

    def embed(
        self,
        actions: torch.Tensor,
        state: torch.Tensor | None,
        state_mask: torch.Tensor | None,
        consequence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = actions.shape[0]
        dt = self.action_in.weight.dtype
        a = self.action_in(actions.to(dt)) + self.type_embed.weight[1]
        if state is None:
            s = self.null_state.expand(B, 1, -1).to(dt)
        else:
            s = self.state_in(state.to(dt))[:, None]
            if state_mask is not None:
                keep = state_mask.any(dim=-1)[:, None, None]
                s = torch.where(keep, s, self.null_state.to(dt))
        s = s + self.type_embed.weight[0]
        toks = [s, a]
        if self.cfg.consequence_dim > 0:
            if consequence is None:
                raise ValueError("consequence tokens configured but no consequence input given")
            if consequence.shape[1] != self.consequence_steps:
                raise ValueError(
                    f"expected {self.consequence_steps} consequence steps, got {consequence.shape[1]}"
                )
            toks.append(self.consequence_in(consequence.to(dt)) + self.type_embed.weight[2])
        return torch.cat(toks, dim=1)

    def positions(self, device) -> torch.Tensor:
        A = self.cfg.chunk
        pos = [
            torch.zeros(1, dtype=torch.long, device=device),
            torch.arange(1, A + 1, device=device),
        ]
        if self.consequence_steps:
            pos.append(torch.arange(1, self.consequence_steps + 1, device=device))
        return torch.cat(pos)

    def conditioning(
        self, s_action: torch.Tensor, text: torch.Tensor, s_consequence: torch.Tensor | None = None
    ):
        B = s_action.shape[0]
        A, Hc = (self.cfg.chunk, self.consequence_steps)
        temb_a = self.time(s_action)
        temb_c = temb_a if s_consequence is None or Hc == 0 else self.time(s_consequence)
        tok = torch.cat(
            [temb_a[:, None].expand(B, 1 + A, -1)]
            + ([temb_c[:, None].expand(B, Hc, -1)] if Hc else []),
            dim=1,
        )
        ada = self.time_to_ada(tok).view(B, tok.shape[1], 6, self.cfg.width)
        ctx = self.ctx_in(text.to(self.ctx_in[0].weight.dtype))
        return (tok, ada, ctx)

    def readout(self, x: torch.Tensor, temb_tok: torch.Tensor):
        shift, scale = [
            t.squeeze(2) for t in (self.ada_out[:, None] + temb_tok[:, :, None]).chunk(2, dim=2)
        ]
        h = (layer_norm_fp32(self.norm_out, x).float() * (1 + scale) + shift).type_as(x)
        A = self.cfg.chunk
        v_a = self.action_out(h[:, 1 : 1 + A])
        v_c = (
            self.consequence_out(h[:, 1 + A : 1 + A + self.consequence_steps])
            if self.consequence_steps
            else None
        )
        return (v_a, v_c)
