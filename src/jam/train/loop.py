from contextlib import nullcontext
from itertools import islice

import torch
import torch.distributed as dist


def train_epoch(
    model,
    batches,
    objective,
    optimizer,
    *,
    device,
    grad_accum=1,
    grad_clip=1.0,
    bf16=False,
    generator=None,
):
    if grad_accum < 1:
        raise ValueError("grad_accum must be positive")
    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    if bf16 and device.type != "cuda":
        raise ValueError("bf16 training requires CUDA")
    model.train()
    iterator = iter(batches)
    totals, updates = {}, 0
    while group := list(islice(iterator, grad_accum)):
        optimizer.zero_grad(set_to_none=True)
        local = {}
        for index, batch in enumerate(group):
            batch = {key: value.to(device) for key, value in batch.items()}
            sync = (
                model.no_sync()
                if index < len(group) - 1 and hasattr(model, "no_sync")
                else nullcontext()
            )
            with sync:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                    result = objective(model, batch, generator=generator)
                    loss = result["loss"] / len(group)
                loss.backward()
            for key, value in result["logs"].items():
                local[key] = local.get(key, 0.0) + value / len(group)
        clip = (
            model.clip_grad_norm_
            if hasattr(model, "clip_grad_norm_")
            else lambda value: torch.nn.utils.clip_grad_norm_(model.parameters(), value)
        )
        norm = clip(grad_clip)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite gradient norm")
        optimizer.step()
        updates += 1
        for key, value in local.items():
            totals[key] = totals.get(key, 0.0) + value.detach()
    if updates == 0:
        raise ValueError("no prepared batches were supplied")
    metrics = {}
    for key, total in totals.items():
        value = total / updates
        if dist.is_initialized():
            dist.all_reduce(value)
            value /= dist.get_world_size()
        metrics[key] = float(value)
    return metrics
