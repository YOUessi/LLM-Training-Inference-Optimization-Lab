"""Repeatable single-host vLLM concurrency sweep.

Runs one benchmark at a time so concurrent sweeps do not contaminate each other.
All numerical summaries come from actual HTTP response streaming, not mocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8026")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", default="1,2,4",
                        help="Comma-separated concurrency levels, sequentially executed")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/vllm_sweep"))
    args = parser.parse_args(argv)
    try:
        values = [int(x.strip()) for x in args.concurrency.split(",")]
    except ValueError:
        parser.error("--concurrency requires positive integer comma-separated values")
    if not values or any(v <= 0 for v in values) or len(set(values)) != len(values):
        parser.error("--concurrency must list distinct positive integers")
    if any(v > 64 for v in values):
        parser.error("--concurrency max is 64 to bound workload")
    args.concurrency = tuple(values)
    if args.requests <= 0 or args.warmup_requests < 0 or args.max_tokens <= 0:
        parser.error("requests and max tokens must be positive; warmup must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    from llm_opt_lab.serving.bench_client import main as bench
    cases = []
    for c in a.concurrency:
        output = a.output_dir / f"concurrency_{c}.json"
        code = bench([
            "--base-url", a.base_url,
            "--model", a.model,
            "--requests", str(a.requests),
            "--warmup-requests", str(a.warmup_requests),
            "--concurrency", str(c),
            "--max-tokens", str(a.max_tokens),
            "--seed", str(a.seed),
            "--output", str(output),
        ])
        row = {"concurrency": c, "returncode": code, "raw_file": str(output)}
        if output.is_file():
            data = json.loads(output.read_text(encoding="utf-8"))
            row["status"] = data["status"]
            row["summary"] = data["summary"]
        else:
            row["status"] = "blocked"
        cases.append(row)
        if code != 0:
            # No misleading "completed sweep" if an endpoint fails mid-run.
            break
    success = len(cases) == len(a.concurrency) and all(c["returncode"] == 0 for c in cases)
    write_json_atomic(a.output_dir / "summary.json", {
        "status": "measured" if success else "partial_or_blocked",
        "kind": "vllm_concurrency_sweep",
        "environment": environment(),
        "model": a.model,
        "workload": {
            "concurrency": a.concurrency,
            "requests_per_case": a.requests,
            "warmup_per_case": a.warmup_requests,
            "max_output_tokens": a.max_tokens,
            "seed": a.seed,
            "request_prompts": "fixed template plus per-index task ID",
        },
        "cases": cases,
    })
    for c in cases:
        summary = c.get("summary", {})
        print(f"concurrency={c['concurrency']} status={c['status']} "
              f"p95_ttft_ms={summary.get('p95_ttft_ms')} "
              f"req/s={summary.get('requests_per_second')} "
              f"token/s={summary.get('output_tokens_per_second')}")
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
