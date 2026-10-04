import functools
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def initialize():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
        return device, dist.get_rank(), dist.get_world_size()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1


def wrap_model(model, device, *, bf16=False):
    if not dist.is_initialized():
        return model.to(device)
    # Agent blocks are called in pieces, so their parameters remain in the root unit.
    block_types = {type(block) for block in model.world.net.blocks}
    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=block_types)
    precision = (
        MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16
        )
        if bf16
        else None
    )
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=precision,
        device_id=device,
        device_mesh=init_device_mesh(device.type, (dist.get_world_size(),)),
        use_orig_params=True,
        limit_all_gathers=True,
    )
