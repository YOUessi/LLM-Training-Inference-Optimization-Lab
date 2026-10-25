"""Torch operator/kernel profiling while Nsight is unavailable.

This is *not* a replacement for Nsight Compute hardware occupancy or memory
throughput counters. Excludes torch.compile compilation from measured steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=256)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--num-warps", type=int, choices=(1, 2, 4, 8, 16), default=8)
    p.add_argument("--output", type=Path, default=Path("results/torch_profiler_rmsnorm.json"))
    a = p.parse_args(argv)
    if a.rows <= 0 or a.cols <= 0 or a.cols > 16384 or a.steps <= 0 or a.warmup < 1:
        p.error("rows>0, 1<=cols<=16384, steps>0, warmup>=1")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        import torch
        from torch.profiler import profile, ProfilerActivity
        from llm_opt_lab.kernels.triton_ops import rmsnorm, rmsnorm_grouped
    except ImportError as exc:
        print(f"BLOCKED: torch with CUDA and triton required ({exc})", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("BLOCKED: CUDA profiler needs NVIDIA GPU", file=sys.stderr)
        return 2

    try:
        torch.manual_seed(3571)
        x = torch.randn((a.rows, a.cols), device="cuda", dtype=getattr(torch, a.dtype))
        eps = 1e-6

        def baseline(t):
            return (t * torch.rsqrt(t.float().square().mean(dim=-1, keepdim=True) + eps)).to(t.dtype)

        compiled = torch.compile(baseline, backend="inductor", fullgraph=True)
        expected = baseline(x)
        atol = 3e-3 if a.dtype == "float16" else 2e-5
        rtol = atol
        variants = {
            "torch_inductor": lambda: compiled(x),
            "triton_original": lambda: rmsnorm(x, eps),
            "triton_row1_warps": lambda: rmsnorm_grouped(x, eps, rows_per_program=1, num_warps=a.num_warps),
        }
        rows = {}
        for name, fn in variants.items():
            torch.testing.assert_close(fn(), expected, atol=atol, rtol=rtol)
            for _ in range(a.warmup):
                fn()
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                for _ in range(a.steps):
                    fn()
                torch.cuda.synchronize()
            records = []
            for item in prof.key_averages():
                # Torch versions expose either self_device_time_total or self_cuda_time_total.
                self_gpu_us = float(getattr(item, "self_device_time_total",
                                  getattr(item, "self_cuda_time_total", 0.0)))
                device_total_us = float(getattr(item, "device_time_total",
                                      getattr(item, "cuda_time_total", 0.0)))
                records.append({
                    "operator": str(item.key),
                    "call_count": int(item.count),
                    "self_device_time_us": self_gpu_us,
                    "total_device_time_us": device_total_us,
                    "self_cpu_time_us": float(item.self_cpu_time_total),
                })
            rows[name] = {
                "top_by_self_device_time": sorted(
                    records, key=lambda r: r["self_device_time_us"], reverse=True
                )[:15],
                "operator_count": len(records),
            }

        write_json_atomic(a.output, {
            "status": "measured", "kind": "torch_profiler_kernel_diagnostic",
            "not_nsight_hardware_counters": True,
            "configuration": {
                "rows": a.rows, "cols": a.cols, "dtype": a.dtype,
                "steps": a.steps, "warmup": a.warmup, "num_warps": a.num_warps
            },
            "environment": environment(),
            "profile": rows,
        })
        for name, row in rows.items():
            top = row["top_by_self_device_time"][:4]
            print(name, [(x["operator"], round(x["self_device_time_us"], 2)) for x in top])
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
