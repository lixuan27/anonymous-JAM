from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .action_stream import AgentStream, rope_1d
from .coupling import TokenBook, build_joint_mask
from .world_stream import LayerHub, WorldStream


@dataclass
class JamOutput:
    v_world: torch.Tensor
    v_action: torch.Tensor
    v_consequence: torch.Tensor | None


def pair_layers(n_world: int, n_agent: int) -> dict[int, int]:
    if not 1 <= n_agent <= n_world:
        raise ValueError("agent depth must be between one and world depth")
    if n_agent == n_world:
        return {i: i for i in range(n_world)}
    out = {}
    for j in range(n_agent):
        i = int(round((j + 0.5) * n_world / n_agent - 0.5))
        out[min(max(i, 0), n_world - 1)] = j
    return out


class JointActionWorldModel(nn.Module):
    def __init__(
        self,
        world: WorldStream,
        agent: AgentStream,
        *,
        gradient_checkpointing: bool = False,
        clean_frames: int = 1,
    ):
        super().__init__()
        if agent.cfg.heads != world.heads or agent.cfg.head_dim != world.head_dim:
            raise ValueError(
                f"agent heads/head_dim ({agent.cfg.heads}x{agent.cfg.head_dim}) must match the world backbone ({world.heads}x{world.head_dim}) to share one attention space"
            )
        self.world = world
        self.agent = agent
        self.hub: LayerHub = world.hub
        self.gradient_checkpointing = gradient_checkpointing
        self.clean_frames = clean_frames
        self.layer_map = pair_layers(world.num_layers, agent.cfg.layers)

    def _masks(self, Lw: int, clean_tokens: int, La: int, device):
        A = self.agent.cfg.chunk
        n_csq = self.agent.consequence_steps
        book = TokenBook(
            world_tokens=Lw,
            clean_world_tokens=clean_tokens,
            agent_condition_tokens=La - A - n_csq,
            action_tokens=A,
            consequence_tokens=n_csq,
        )
        full = build_joint_mask(book, device=device)
        W, A = (book.world, book.agent)
        world_rows_full = full[W, :][None, None]
        world_rows_only = full[W, W][None, None]
        agent_rows_full = full[A, :][None, None]
        return {
            "world_full": world_rows_full,
            "world_only": None if bool(world_rows_only.all()) else world_rows_only,
            "agent_full": agent_rows_full,
        }

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        s_world: torch.Tensor,
        s_action: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        consequence: torch.Tensor | None = None,
        s_consequence: torch.Tensor | None = None,
    ) -> JamOutput:
        hub = self.hub
        hub.reset()
        wst = self.world.begin(
            latents, s_world, text, clean_frames=self.clean_frames, text_mask=text_mask
        )
        Lw, clean_tokens = (wst["x"].shape[1], wst["clean_tokens"])
        x_a = self.agent.embed(actions, state, state_mask, consequence)
        La = x_a.shape[1]
        temb_a, ada, ctx_a = self.agent.conditioning(s_action, text, s_consequence)
        pos = self.agent.positions(x_a.device)
        cos, sin = rope_1d(pos, self.agent.cfg.head_dim)
        masks = self._masks(Lw, clean_tokens, La, x_a.device)

        def run_layer(i: int, x_w: torch.Tensor, x_a: torch.Tensor):
            wst["x"] = x_w
            j = self.layer_map.get(i)
            world_reads = j is not None
            if j is not None:
                blk = self.agent.blocks[j]
                sh, sc, g, csh, csc, cg = blk.modulation(ada)
                q_a, k_a, v_a = blk.project_qkv(x_a, sh, sc, cos, sin)
                hub.agent_k, hub.agent_v = k_a, v_a
            hub.world_reads_agent = world_reads
            hub.world_query_mask = masks["world_full"] if world_reads else masks["world_only"]
            self.world.layer(i, wst)
            x_w = wst["x"]
            if j is not None:
                k_w, v_w = hub.world_k, hub.world_v
                keys = torch.cat([k_w.to(k_a.dtype), k_a], dim=1)
                values = torch.cat([v_w.to(v_a.dtype), v_a], dim=1)
                mask = masks["agent_full"]
                out = F.scaled_dot_product_attention(
                    q_a.transpose(1, 2),
                    keys.transpose(1, 2),
                    values.transpose(1, 2),
                    attn_mask=mask,
                )
                out = out.transpose(1, 2).reshape(x_a.shape[0], La, -1)
                x_a = blk.finish(x_a, out, g, csh, csc, cg, ctx_a, text_mask)
            return (x_w, x_a)

        x_w = wst["x"]
        for i in range(self.world.num_layers):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x_w, x_a = checkpoint(run_layer, i, x_w, x_a, use_reentrant=False)
            else:
                x_w, x_a = run_layer(i, x_w, x_a)
        wst["x"] = x_w
        v_world = self.world.finish(wst, latents.shape)
        v_action, v_consequence = self.agent.readout(x_a, temb_a)
        return JamOutput(v_world, v_action, v_consequence)
