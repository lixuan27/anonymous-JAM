import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from jam.model.build import build_model
from jam.model.objective import JointFlowObjective, ObjectiveConfig

from .batch import validate_batch
from .checkpoint import load_checkpoint, save_checkpoint
from .distributed import initialize, wrap_model
from .loop import train_epoch
from .optim import OptimConfig, build_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--lr-agent", type=float)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    device, rank, world_size = initialize()
    torch.manual_seed(args.seed)
    cfg = json.loads(Path(args.config).read_text())
    model = build_model(
        args.backbone, cfg["agent"], gradient_checkpointing=args.gradient_checkpointing
    )
    agent_config = model.agent.cfg
    model = wrap_model(model, device, bf16=args.bf16)
    optimizer = build_optimizer(
        model, OptimConfig(args.lr, args.lr_agent, grad_clip=args.grad_clip)
    )
    objective = JointFlowObjective(ObjectiveConfig(**cfg.get("objective", {})))
    files = sorted(Path(args.batches).glob("*.safetensors"))
    if len(files) < world_size or len(files) % world_size:
        raise ValueError("prepared batch count must be positive and divisible by process count")
    files = files[rank::world_size]
    torch.manual_seed(args.seed + rank)
    start = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    for epoch in range(start, args.epochs):
        metrics = train_epoch(
            model,
            (validate_batch(load_file(str(path)), agent_config) for path in files),
            objective,
            optimizer,
            device=device,
            grad_accum=args.grad_accum,
            grad_clip=args.grad_clip,
            bf16=args.bf16,
        )
        save_checkpoint(
            Path(args.output) / f"epoch-{epoch + 1}", model, optimizer, completed_epochs=epoch + 1
        )
        if rank == 0:
            print(json.dumps(metrics), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
