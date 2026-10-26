"""Real PyTorch FSDP1 forward/backward/optimizer infrastructure smoke.

A world_size=1 run intentionally demonstrates API correctness only: PyTorch
commonly falls back to NO_SHARD, so it CANNOT prove memory savings,
sharded optimizer behavior or multi-GPU scalability.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--features", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=Path, default=Path("results/fsdp_smoke.json"))
    a = p.parse_args(argv)
    if min(a.steps, a.batch_size, a.features, a.hidden) <= 0 or a.lr <= 0:
        p.error("steps/batch-size/features/hidden/lr must be positive")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        import torch
        import torch.distributed as dist
        from torch import nn
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError as exc:
        print(f"BLOCKED: PyTorch with FSDP required: {exc}", file=sys.stderr)
        return 2

    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    if not all(k in os.environ for k in keys):
        print("BLOCKED: must use torchrun", file=sys.stderr)
        return 2
    rank, world_size, local_rank = (int(os.environ[k]) for k in keys[:3])
    if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
        print("BLOCKED: one actual NVIDIA GPU per local rank required", file=sys.stderr)
        return 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    try:
        torch.manual_seed(a.seed)
        module = nn.Sequential(
            nn.Linear(a.features, a.hidden), nn.GELU(),
            nn.Linear(a.hidden, a.features),
        ).to(device)
        fsdp = FSDP(module, device_id=device, use_orig_params=True)
        optimizer = torch.optim.AdamW(fsdp.parameters(), lr=a.lr)
        data_gen = torch.Generator(device=device)
        data_gen.manual_seed(a.seed + 10000 + rank)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        runtimes: list[float] = []
        losses: list[float] = []
        for _ in range(a.steps):
            x = torch.randn(
                (a.batch_size, a.features), device=device, generator=data_gen
            )
            targets = 0.5 * x
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            pred = fsdp(x)
            loss = (pred - targets).square().mean()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            runtimes.append(time.perf_counter() - started)
            losses.append(float(loss.detach().item()))

        slowest = torch.tensor(sum(runtimes), device=device, dtype=torch.float64)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
        if rank == 0:
            output = {
                "status": "measured",
                "kind": "synthetic_fsdp1_training_smoke",
                "not_an_llm_training_result": True,
                "not_multigpu_scaling_evidence": world_size == 1,
                "environment": environment(),
                "config": {
                    "steps": a.steps, "world_size": world_size,
                    "batch_size_per_rank": a.batch_size,
                    "features": a.features, "hidden": a.hidden,
                    "seed": a.seed, "lr": a.lr,
                    "requested_parallelism": "FSDP1",
                    "effective_sharding_strategy": str(fsdp.sharding_strategy),
                },
                "results": {
                    "final_loss": losses[-1],
                    "first_loss": losses[0],
                    "median_step_ms": statistics.median(runtimes) * 1000.,
                    "slowest_rank_total_step_s": float(slowest.item()),
                    "throughput_samples_per_second": (
                        world_size * a.steps * a.batch_size / float(slowest.item())
                        if slowest.item() > 0 else None
                    ),
                    "peak_allocated_gpu_gib": (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    ),
                    "peak_reserved_gpu_gib": (
                        torch.cuda.max_memory_reserved(device) / 1024**3
                    ),
                },
            }
            write_json_atomic(a.output, output)
            print(
                "FSDP smoke measured: world_size=", world_size,
                "effective=", str(fsdp.sharding_strategy),
                "median_step_ms=", round(output["results"]["median_step_ms"], 3),
            )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
