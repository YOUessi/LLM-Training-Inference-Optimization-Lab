"""Reproduce exact single- or multi-rank DDP training resume.

Run uninterrupted total steps versus a split + restart, with independent
torchrun processes. Checks exact model, optimizer AND per-rank RNG states.
This is a synthetic MLP systems test, not LLM/FSDP evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from llm_opt_lab.provenance import environment, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--split", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--features", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    if a.world_size < 1 or a.batch_size < 1 or min(a.features, a.hidden) < 1:
        p.error("world-size/batch-size/features/hidden must be positive")
    if not 1 <= a.split < a.steps:
        p.error("require 1 <= split < steps")
    return a


def exact_state(x: Any, y: Any, torch) -> bool:
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and torch.equal(x.cpu(), y.cpu())
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(
            exact_state(x[k], y[k], torch) for k in x
        )
    if isinstance(x, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(
            exact_state(v, w, torch) for v, w in zip(x, y)
        )
    return x == y


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        import torch
    except ImportError:
        print("BLOCKED: PyTorch required to test DDP restart", file=sys.stderr)
        return 2
    if a.device == "cuda" and torch.cuda.device_count() < a.world_size:
        print("BLOCKED: insufficient visible NVIDIA GPUs", file=sys.stderr)
        return 2
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        print("BLOCKED: output dir must be empty to avoid stale checkpoints", file=sys.stderr)
        return 2
    a.output_dir.mkdir(parents=True, exist_ok=True)

    base = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nnodes=1", f"--nproc_per_node={a.world_size}",
        "-m", "llm_opt_lab.training.ddp_train",
        "--device", a.device,
        "--batch-size", str(a.batch_size),
        "--features", str(a.features),
        "--hidden", str(a.hidden),
        "--seed", str(a.seed),
    ]
    jobs = [
        ("full", ["--steps", str(a.steps)]),
        ("part", ["--steps", str(a.split)]),
        ("resumed", ["--steps", str(a.steps), "--resume", str(a.output_dir/"part.pt")]),
    ]
    completed = []
    for label, extras in jobs:
        argv_run = base + extras + [
            "--checkpoint", str(a.output_dir/(label+".pt")),
            "--output", str(a.output_dir/(label+".json")),
        ]
        r = subprocess.run(argv_run, capture_output=True, text=True, timeout=240, check=False)
        # Logs remain local, including negative results. Avoid copying them
        # to the public repo without review for sensitive context.
        (a.output_dir/(label+".log")).write_text(
            r.stdout + "\n--- STDERR ---\n" + r.stderr,
            encoding="utf-8",
        )
        completed.append({"name": label, "returncode": r.returncode})
        if r.returncode:
            break
    report: dict[str, Any] = {
        "status": "failed",
        "kind": "ddp_exact_resume_validation",
        "not_an_llm_training_result": True,
        "environment": environment(),
        "configuration": {
            "device": a.device, "world_size": a.world_size,
            "steps": a.steps, "split": a.split, "batch_size": a.batch_size,
            "features": a.features, "hidden": a.hidden, "seed": a.seed,
        },
        "subprocesses": completed,
        "equality": None,
    }
    if len(completed) != 3 or any(j["returncode"] for j in completed):
        write_json_atomic(a.output_dir/"validation.json", report)
        print("FAILED: one of the three real torchrun processes failed", file=sys.stderr)
        return 1

    full = torch.load(a.output_dir/"full.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(a.output_dir/"resumed.pt", map_location="cpu", weights_only=True)
    records = [
        json.loads((a.output_dir/(label+".json")).read_text())
        for label in ("full", "part", "resumed")
    ]
    equality = {
        "model": exact_state(full["model"], resumed["model"], torch),
        "optimizer": exact_state(full["optimizer"], resumed["optimizer"], torch),
        "per_rank_rng_states": exact_state(
            full["rank_rng_states"], resumed["rank_rng_states"], torch
        ),
        "model_sha256": (
            records[0]["results"]["model_sha256"]
            == records[2]["results"]["model_sha256"]
        ),
    }
    report["equality"] = equality
    report["model_sha256_full"] = records[0]["results"]["model_sha256"]
    report["model_sha256_resumed"] = records[2]["results"]["model_sha256"]
    report["checkpoint_version"] = full.get("checkpoint_version")
    report["status"] = "pass" if all(equality.values()) else "failed"
    write_json_atomic(a.output_dir/"validation.json", report)
    print(f"{a.device}/{a.world_size} DDP exact resume: {report['status']} {equality}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
