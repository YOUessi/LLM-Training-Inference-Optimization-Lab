"""Tune grouped RMSNorm Triton kernels on a real NVIDIA GPU.

This is a candidate-discovery experiment, NOT a proven optimized dispatch
policy. Always compare with a correctly fused torch.compile baseline.
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from llm_opt_lab.provenance import environment, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warps", default="1,2,4,8")
    parser.add_argument("--rows-per-program", default="1,2,4,8")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--rep-ms", type=int, default=35)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/rmsnorm_tune.json"))
    args = parser.parse_args(argv)
    if args.rows <= 0 or not 1 <= args.cols <= 16384 or args.trials < 2 or args.rep_ms <= 0:
        parser.error("rows>0, cols 1..16384, trials>=2, rep-ms>0")
    if not math.isfinite(args.eps) or args.eps <= 0:
        parser.error("eps must be finite and positive")
    for name in ("warps", "rows_per_program"):
        values = [int(x) for x in getattr(args, name).split(",") if x.strip()]
        if not values or any(v not in (1, 2, 4, 8, 16) for v in values):
            parser.error(f"{name} must list powers of two from 1 to 16")
        setattr(args, name, tuple(sorted(set(values))))
    return args


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import torch
        import triton
        import triton.testing
    except ImportError as exc:
        print(f"BLOCKED: requires CUDA PyTorch and Triton ({exc})", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("BLOCKED: this is a real CUDA benchmarking tool, not a CPU simulation", file=sys.stderr)
        return 2

    try:
        from llm_opt_lab.kernels import triton_ops
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        x = torch.randn(
            (args.rows, args.cols), device="cuda", dtype=getattr(torch, args.dtype)
        )

        def torch_rmsnorm(t):
            return (
                t * torch.rsqrt(t.float().square().mean(dim=-1, keepdim=True) + args.eps)
            ).to(t.dtype)

        expected = torch_rmsnorm(x)
        atol, rtol = (3e-3, 3e-3) if args.dtype == "float16" else (2e-5, 2e-5)
        compiled = torch.compile(torch_rmsnorm, backend="inductor", fullgraph=True)
        torch.testing.assert_close(compiled(x), expected, atol=atol, rtol=rtol)

        # CUDA graph replay minimizes CPU launch overhead for microsecond kernels.
        def time_once(fn) -> float:
            return float(triton.testing.do_bench_cudagraph(fn, rep=args.rep_ms, return_mode="median"))

        candidates: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for rows_per_program in args.rows_per_program:
            for warps in args.warps:
                # Do not attempt configurations that are likely to blow register/shared resource budgets.
                elements = triton.next_power_of_2(args.cols) * rows_per_program
                if elements > 32768:
                    continue
                def fn(rpp=rows_per_program, w=warps):
                    return triton_ops.rmsnorm_grouped(
                        x, args.eps, rows_per_program=rpp, num_warps=w
                    )
                try:
                    result = fn()
                    torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)
                    max_abs = (result.float() - expected.float()).abs().max().item()
                    candidates.append({
                        "rows_per_program": rows_per_program,
                        "num_warps": warps,
                        "max_absolute_error": float(max_abs),
                        "latency_ms": [],
                        "correctness": "pass",
                    })
                except Exception as exc:
                    failures.append({
                        "rows_per_program": rows_per_program,
                        "num_warps": warps,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:240],
                    })

        if not candidates:
            print("FAILED: all candidate configurations rejected or failed correctness", file=sys.stderr)
            return 1

        rand = random.Random(args.seed)
        raw_baselines: dict[str, list[float]] = {
            "compiled_torch_ms": [], "legacy_triton_ms": [],
        }
        for _ in range(args.trials):
            # Interleave baselines and candidates; eliminate obvious warmup/order bias.
            actions = [("compiled", None), ("legacy", None)] + [
                ("candidate", i) for i in range(len(candidates))
            ]
            rand.shuffle(actions)
            for kind, candidate_index in actions:
                if kind == "compiled":
                    raw_baselines["compiled_torch_ms"].append(time_once(lambda: compiled(x)))
                elif kind == "legacy":
                    raw_baselines["legacy_triton_ms"].append(
                        time_once(lambda: triton_ops.rmsnorm(x, args.eps))
                    )
                else:
                    c = candidates[candidate_index]
                    c["latency_ms"].append(time_once(
                        lambda c=c: triton_ops.rmsnorm_grouped(
                            x, args.eps,
                            rows_per_program=c["rows_per_program"],
                            num_warps=c["num_warps"],
                        )
                    ))

        # Complete results are recorded, not just cherry-picked best configurations.
        for c in candidates:
            c["median_ms"] = median(c["latency_ms"])
            c["p95_across_trials_ms"] = sorted(c["latency_ms"])[
                math.ceil(0.95 * len(c["latency_ms"])) - 1
            ]
        candidates.sort(key=lambda c: c["median_ms"])
        compiled_ms = median(raw_baselines["compiled_torch_ms"])
        legacy_ms = median(raw_baselines["legacy_triton_ms"])
        best = candidates[0]
        payload = {
            "status": "measured",
            "kind": "rmsnorm_configuration_sweep",
            "experimental_not_deployed": True,
            "environment": environment(),
            "triton_version": triton.__version__,
            "tensor": {
                "rows": args.rows, "cols": args.cols, "dtype": args.dtype, "eps": args.eps
            },
            "parameters": {
                "seed": args.seed, "trials": args.trials, "rep_ms": args.rep_ms,
                "num_warps": args.warps, "rows_per_program": args.rows_per_program,
                "timing_method": "triton.testing.do_bench_cudagraph",
            },
            "baselines": {
                "compiled_torch_ms_by_trial": raw_baselines["compiled_torch_ms"],
                "legacy_triton_ms_by_trial": raw_baselines["legacy_triton_ms"],
                "compiled_torch_median_ms": compiled_ms,
                "legacy_triton_median_ms": legacy_ms,
            },
            "candidates": candidates,
            "failures": failures,
            "best_in_sample": {
                "rows_per_program": best["rows_per_program"],
                "num_warps": best["num_warps"],
                "median_ms": best["median_ms"],
                "speedup_vs_compiled": compiled_ms / best["median_ms"],
                "speedup_vs_legacy": legacy_ms / best["median_ms"],
            },
        }
        write_json_atomic(args.output, payload)
        print(
            f"TUNED shape={args.rows}x{args.cols} {args.dtype}: "
            f"{len(candidates)} candidates, {len(failures)} failures; "
            f"best grouped rows={best['rows_per_program']}, warps={best['num_warps']}; "
            f"median_ms={best['median_ms']:.6f}, "
            f"speedup_vs_compiled={compiled_ms / best['median_ms']:.3f}x"
        )
        return 0
    except Exception as exc:
        print(f"FAILED: tuner error {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
