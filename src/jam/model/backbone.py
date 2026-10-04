import json
from pathlib import Path

import diffusers
from diffusers.models.modeling_utils import ModelMixin


def read_backbone_config(path):
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("backbone configuration must be a JSON object")
    return config


def resolve_backbone_type(config):
    name = config.get("_class_name")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("checkpoint configuration needs a valid model class")
    model_type = getattr(diffusers, name, None)
    if not isinstance(model_type, type) or not issubclass(model_type, ModelMixin):
        raise ValueError("checkpoint model class must belong to the installed backbone library")
    return model_type


def check_backbone_interface(backbone):
    required = (
        "blocks",
        "patch_embedding",
        "condition_embedder",
        "rope",
        "scale_shift_table",
        "norm_out",
        "proj_out",
    )
    if any(not hasattr(backbone, name) for name in required) or not backbone.blocks:
        raise ValueError("backbone does not implement the required video-transformer interface")
    if backbone.config.patch_size[0] != 1:
        raise ValueError("the observed prefix requires temporal patch size one")
    return backbone
