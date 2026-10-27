"""Targeted, NVTX-labeled real GPU RMSNorm execution.

No synthetic performance numbers. One deliberately fixed tensor is passed
to an eager Torch baseline, a compiled Torch fusion, and custom Triton.
The command emits trace labels and correctness results; use Nsight Systems
for *instrumented GPU activity*, separate from uninstrumented speed tests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/nvtx_rmsnorm.json"))
    args = parser.parse_args(argv)
    if args.rows < 1 or not 1 <= args.cols <= 16384:
        parser.error("rows>=1 and columns 1..16384 required")
    if args.warmup < 1 or args.repeats < 1 or not 0 < args.eps < float("inf"):
        parser.error("warmup/repeats positive, eps finite positive")
    return args


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        import torch
        from llm_opt_lab.kernels.triton_ops import rmsnorm
    except ImportError as exc:
        print(f"BLOCKED: NVIDIA PyTorch and Triton required: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("BLOCKED: CUDA GPU unavailable", file=sys.stderr)
        return 2
    try:
        torch.manual_seed(a.seed)
        x = torch.randn(
            (a.rows, a.cols), device="cuda", dtype=getattr(torch, a.dtype)
        )

        def eager(t):
            return (
                t * torch.rsqrt(
                    t.float().square().mean(dim=-1, keepdim=True) + a.eps
                )
            ).to(t.dtype)

        compiled = torch.compile(eager, backend="inductor", fullgraph=True)
        cases = {
            "eager_unfused": eager,
            "torch_inductor_fused": compiled,
            "custom_triton": lambda t: rmsnorm(t, eps=a.eps),
        }
        expected = eager(x)
        tolerance = 3e-3 if a.dtype == "float16" else 2e-5
        records = {}
        for name, fn in cases.items():
            result = fn(x)
            torch.testing.assert_close(
                result, expected, atol=tolerance, rtol=tolerance
            )
            max_error = float(
                (result.float() - expected.float()).abs().max().item()
            )
            for _ in range(a.warmup):
                fn(x)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push("llm-opt:" + name)
            try:
                for _ in range(a.repeats):
                    fn(x)
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            records[name] = {
                "correctness": "pass", "max_absolute_error": max_error,
                "nvtx_range": "llm-opt:" + name,
                "iterations": a.repeats,
            }
        write_json_atomic(a.output, {
            "status": "measured",
            "kind": "real_cuda_rmsnorm_nvtx_trace_driver",
            "not_an_uninstrumented_speed_benchmark": True,
            "environment": environment(),
            "config": {
                "rows": a.rows, "cols": a.cols, "dtype": a.dtype,
                "warmup": a.warmup, "repeats": a.repeats,
                "seed": a.seed, "eps": a.eps,
            },
            "cases": records,
        })
        for name, info in records.items():
            print(
                name, "correctness:", info["correctness"],
                "maximum absolute error:", info["max_absolute_error"],
                "NVTX range:", info["nvtx_range"],
            )
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
