from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import check_backbone_interface, read_backbone_config, resolve_backbone_type


@dataclass
class LayerHub:
    agent_k: torch.Tensor | None = None
    agent_v: torch.Tensor | None = None
    world_query_mask: torch.Tensor | None = None
    text_mask: torch.Tensor | None = None
    world_reads_agent: bool = False
    world_k: torch.Tensor | None = None
    world_v: torch.Tensor | None = None

    def reset(self) -> None:
        self.agent_k = self.agent_v = None
        self.world_query_mask = None
        self.text_mask = None
        self.world_reads_agent = False
        self.world_k = self.world_v = None


def _rotate(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(x)


class HubAttnProcessor:
    def __init__(self, hub: LayerHub, layer: int):
        self.hub = hub
        self.layer = layer

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None
    ):
        if encoder_hidden_states is not None:
            raise RuntimeError("HubAttnProcessor is a self-attention processor")
        q = attn.norm_q(attn.to_q(hidden_states))
        k = attn.norm_k(attn.to_k(hidden_states))
        v = attn.to_v(hidden_states)
        q = q.unflatten(2, (attn.heads, -1))
        k = k.unflatten(2, (attn.heads, -1))
        v = v.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            q = _rotate(q, *rotary_emb)
            k = _rotate(k, *rotary_emb)
        hub = self.hub
        hub.world_k, hub.world_v = (k, v)
        keys, values = (k, v)
        if hub.world_reads_agent and hub.agent_k is not None:
            keys = torch.cat([k, hub.agent_k.to(k.dtype)], dim=1)
            values = torch.cat([v, hub.agent_v.to(v.dtype)], dim=1)
        mask = hub.world_query_mask
        if mask is not None and mask.shape[-1] != keys.shape[1]:
            mask = mask[..., : keys.shape[1]]
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=mask
        )
        out = out.transpose(1, 2).flatten(2, 3).type_as(q)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


class MaskedContextProcessor:
    def __init__(self, processor, hub: LayerHub):
        self.processor = processor
        self.hub = hub

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None
    ):
        mask = self.hub.text_mask
        if mask is not None:
            if attn.add_k_proj is not None:
                raise ValueError("text masking requires text-only context")
            mask = mask if attention_mask is None else mask & attention_mask
        else:
            mask = attention_mask
        return self.processor(attn, hidden_states, encoder_hidden_states, mask, rotary_emb)


class WorldStream(nn.Module):
    def __init__(self, transformer: nn.Module, hub: LayerHub):
        super().__init__()
        self.net = check_backbone_interface(transformer)
        self.hub = hub
        for i, block in enumerate(self.net.blocks):
            block.attn1.set_processor(HubAttnProcessor(hub, i))
            block.attn2.set_processor(MaskedContextProcessor(block.attn2.processor, hub))

    @property
    def num_layers(self) -> int:
        return len(self.net.blocks)

    @property
    def heads(self) -> int:
        return int(self.net.config.num_attention_heads)

    @property
    def head_dim(self) -> int:
        return int(self.net.config.attention_head_dim)

    @property
    def width(self) -> int:
        return self.heads * self.head_dim

    def token_grid(self, latents: torch.Tensor) -> tuple[int, int, int]:
        _, _, T, H, W = latents.shape
        pt, ph, pw = self.net.config.patch_size
        return (T // pt, H // ph, W // pw)

    def tokens_per_frame(self, latents: torch.Tensor) -> int:
        _, h, w = self.token_grid(latents)
        return h * w

    def begin(
        self,
        latents: torch.Tensor,
        s_world: torch.Tensor,
        text: torch.Tensor,
        clean_frames: int = 1,
        text_mask: torch.Tensor | None = None,
    ):
        net = self.net
        if text_mask is not None:
            if text_mask.dtype != torch.bool or text_mask.shape != text.shape[:2]:
                raise ValueError("text_mask must be boolean (batch, text_tokens)")
            has_text = text_mask.any(dim=1).all()
            if text_mask.device.type == "cpu":
                if not bool(has_text):
                    raise ValueError("each sample must contain at least one valid text token")
            else:
                torch._assert_async(
                    has_text, "each sample must contain at least one valid text token"
                )
            text_mask = text_mask[:, None, None, :].to(text.device)
        B = latents.shape[0]
        T, h, w = self.token_grid(latents)
        wdt = net.patch_embedding.weight.dtype
        latents = latents.to(wdt)
        text = text.to(net.condition_embedder.text_embedder.linear_1.weight.dtype)
        rotary = net.rope(latents)
        x = net.patch_embedding(latents).flatten(2).transpose(1, 2)
        per_frame = h * w
        t_tok = (s_world.float() * 1000.0)[:, None].expand(B, T).clone()
        t_tok[:, :clean_frames] = 0.0
        t_tok = t_tok.repeat_interleave(per_frame, dim=1)
        temb, tproj, ctx, _ = net.condition_embedder(
            t_tok.flatten(), text, None, timestep_seq_len=t_tok.shape[1]
        )
        tproj = tproj.unflatten(2, (6, -1))
        return {
            "x": x,
            "rotary": rotary,
            "temb": temb,
            "tproj": tproj,
            "ctx": ctx,
            "text_mask": text_mask,
            "grid": (T, h, w),
            "clean_tokens": clean_frames * per_frame,
        }

    def layer(self, i: int, st: dict) -> None:
        self.hub.text_mask = st["text_mask"]
        st["x"] = self.net.blocks[i](st["x"], st["ctx"], st["tproj"], st["rotary"])

    def finish(self, st: dict, latents_shape: torch.Size) -> torch.Tensor:
        net = self.net
        x, temb = (st["x"], st["temb"])
        B = x.shape[0]
        T, h, w = st["grid"]
        pt, ph, pw = net.config.patch_size
        shift, scale = (
            net.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
        ).chunk(2, dim=2)
        shift, scale = (shift.squeeze(2), scale.squeeze(2))
        x = (net.norm_out(x.float()) * (1 + scale) + shift).type_as(x)
        x = net.proj_out(x)
        x = x.reshape(B, T, h, w, pt, ph, pw, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def load_world_backbone(path: str, *, dtype: torch.dtype = torch.float32):
    from pathlib import Path

    config = read_backbone_config(Path(path) / "transformer" / "config.json")
    model_type = resolve_backbone_type(config)
    return model_type.from_pretrained(
        path, subfolder="transformer", torch_dtype=dtype, local_files_only=True
    )
