from dataclasses import replace

from .action_stream import AgentStream, AgentStreamConfig
from .jam import JointActionWorldModel
from .world_stream import LayerHub, WorldStream, load_world_backbone


def build_model(backbone_path, agent_config, *, gradient_checkpointing=False):
    backbone = load_world_backbone(backbone_path)
    world = WorldStream(backbone, LayerHub())
    config = replace(
        AgentStreamConfig(**agent_config),
        heads=world.heads,
        head_dim=world.head_dim,
        ctx_dim=int(backbone.config.text_dim),
    )
    return JointActionWorldModel(
        world, AgentStream(config), gradient_checkpointing=gradient_checkpointing
    )
