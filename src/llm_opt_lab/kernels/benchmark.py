"""Measured PyTorch-vs-Triton kernels. Run only with real NVIDIA CUDA."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ms", type=int, default=150)
    parser.add_argument("--rep-ms", type=int, default=350)
    parser.add_argument("--torch-compile", action="store_true", help="also benchmark correctness-checked torch.compile/Inductor baseline")
    parser.add_argument("--output", type=Path, default=Path("results/kernel_benchmark.json"))
    args = parser.parse_args(argv)
    if args.rows <= 0 or not (1 <= args.cols <= 16384):
        parser.error("rows must be > 0, cols must be 1..16384")
    if args.eps <= 0 or args.warmup_ms <= 0 or args.rep_ms <= 0:
        parser.error("eps/warmup-ms/rep-ms must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import torch
        import triton
        import triton.testing
    except ImportError as exc:
        print(f"BLOCKED: install NVIDIA-compatible torch and triton: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("BLOCKED: no NVIDIA CUDA GPU; refusing to fabricate benchmark metrics", file=sys.stderr)
        return 2
    try:
        from llm_opt_lab.kernels import triton_ops
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        dtype = getattr(torch, args.dtype)
        x = torch.randn((args.rows, args.cols), device="cuda", dtype=dtype)
        def torch_rmsnorm(t):
            return (t * torch.rsqrt(t.float().square().mean(dim=-1, keepdim=True) + args.eps)).to(t.dtype)

        def torch_softmax(t):
            return torch.softmax(t.float(), dim=-1).to(t.dtype)

        cases = {
            "rmsnorm": (torch_rmsnorm, lambda t: triton_ops.rmsnorm(t, args.eps)),
            "softmax": (torch_softmax, triton_ops.softmax),
        }
        # Fairness: baseline outputs are cast to input dtype just like the Triton output.
        results = {}
        atol = 3e-3 if dtype == torch.float16 else 2e-5
        rtol = 3e-3 if dtype == torch.float16 else 2e-5
        for name, (baseline, candidate) in cases.items():
            expected = baseline(x).to(dtype)
            observed = candidate(x)
            max_abs_error = (expected.float() - observed.float()).abs().max().item()
            torch.testing.assert_close(observed, expected, atol=atol, rtol=rtol)
            torch.cuda.synchronize()

            def bench(fn):
                p50, p95 = triton.testing.do_bench(
                    fn, warmup=args.warmup_ms, rep=args.rep_ms,
                    quantiles=[0.5, 0.95],
                )
                return {"median_ms": float(p50), "p95_ms": float(p95)}

            base_t = bench(lambda: baseline(x).to(dtype))
            tri_t = bench(lambda: candidate(x))
            results[name] = {
                "correctness": "pass",
                "atol": atol,
                "rtol": rtol,
                "max_absolute_error": float(max_abs_error),
                "torch": base_t,
                "triton": tri_t,
                "median_speedup": base_t["median_ms"] / tri_t["median_ms"],
            }
            if args.torch_compile:
                # Compile/warmup time is excluded from steady-state GPU timing.
                compiled = torch.compile(baseline, backend="inductor", fullgraph=True)
                compiled_output = compiled(x).to(dtype)
                torch.testing.assert_close(compiled_output, expected, atol=atol, rtol=rtol)
                compiled_t = bench(lambda: compiled(x).to(dtype))
                results[name]["compiled_torch"] = compiled_t
                results[name]["median_speedup_vs_compiled"] = (
                    compiled_t["median_ms"] / tri_t["median_ms"]
                )
        payload = {
            "status": "measured",
            "kind": "gpu_kernel_benchmark",
            "tensor": {"rows": args.rows, "cols": args.cols, "dtype": args.dtype, "eps": args.eps},
            "seed": args.seed,
            "timing": {"warmup_ms": args.warmup_ms, "rep_ms": args.rep_ms},
            "triton_version": triton.__version__,
            "compiled_baseline_requested": bool(args.torch_compile),
            "environment": environment(),
            "cases": results,
        }
        output = write_json_atomic(args.output, payload)
        print(f"Measured {len(results)} real CUDA kernels; results: {output}")
        for name, r in results.items():
            print(f"  {name}: median speedup={r['median_speedup']:.3f}x, max_abs_error={r['max_absolute_error']:.6g}")
        return 0
    except Exception as exc:
        print(f"FAILED: GPU correctness/benchmark did not pass: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
