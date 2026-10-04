import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def save_checkpoint(path, model, optimizer, *, completed_epochs):
    path = Path(path)
    staging = path.with_name(path.name + ".pending")
    exists = [path.exists() or staging.exists() if _rank() == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(exists, src=0)
    if exists[0]:
        raise FileExistsError(path)
    if _rank() == 0:
        staging.mkdir(parents=True)
    _barrier()
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=str(staging))
    rng = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state()
    torch.save(rng, staging / f"rng-{_rank()}.pt")
    _barrier()
    if _rank() == 0:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        (staging / "state.json").write_text(
            json.dumps({"completed_epochs": completed_epochs, "world_size": world_size})
        )
        (staging / "COMPLETE").write_text("ok\n")
        os.rename(staging, path)
    _barrier()


def load_checkpoint(path, model, optimizer):
    path = Path(path)
    if not (path / "COMPLETE").is_file():
        raise ValueError("checkpoint is incomplete")
    state = json.loads((path / "state.json").read_text())
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if state["world_size"] != world_size:
        raise ValueError("resume requires the same process count")
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.load({"model": model_state, "optimizer": optim_state}, checkpoint_id=str(path))
    set_state_dict(model, optimizer, model_state_dict=model_state, optim_state_dict=optim_state)
    rng = torch.load(path / f"rng-{_rank()}.pt", weights_only=True, map_location="cpu")
    torch.set_rng_state(rng["cpu"])
    if "cuda" in rng:
        torch.cuda.set_rng_state(rng["cuda"])
    return state["completed_epochs"]
