"""Reproducible prefix-reuse workload for a live OpenAI-compatible LLM server.

Controlled arms:
- shared: one identical long document prefix + a short per-request suffix.
- distinct: distinct ID *before* the long document, preventing reusable block
  hash chains within a case and between differently-seeded cases.

Cache enabled/disabled is configured in the SERVER launcher (not here);
this workload records the declared server arm and real prompt_tokens/TTFT.
No fake cache-hit or performance numbers are generated.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_opt_lab.provenance import environment, write_json_atomic
from llm_opt_lab.serving.bench_client import (
    _endpoint, _request, summarize,
)

TECHNICAL_PARAGRAPH = (
    "In a production transformer serving cluster, profile prefill compute, "
    "KV cache reuse, memory bandwidth, request scheduling, GPU occupancy, "
    "and tail latency under controlled workload changes. "
)
RESPONSE_INSTRUCTION = "Answer in one short sentence."


def make_prompt(index: int, *, seed: int, repetitions: int, mode: str) -> str:
    """Keep an equal-length context while varying the first prefill block.

    The first content token differs across requests in the 'distinct' arm.
    Absolute prompt length is *measured by server usage*, not estimated here.
    """
    if not 1 <= repetitions <= 60:
        raise ValueError("repetitions must be between 1 and 60")
    if mode not in ("shared", "distinct"):
        raise ValueError("mode must be shared or distinct")
    marker = "V00000000" if mode == "shared" else f"V{(seed * 100000 + index) % (1 << 32):08x}"
    return marker + " " + TECHNICAL_PARAGRAPH * repetitions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8026")
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=("shared", "distinct"), required=True)
    p.add_argument("--declared-cache-state", choices=("on", "off"), required=True)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--requests", type=int, default=24)
    p.add_argument("--warmup-requests", type=int, default=2)
    p.add_argument("--prefix-repetitions", type=int, default=35)
    p.add_argument("--min-prompt-tokens", type=int, default=900)
    p.add_argument("--max-prompt-tokens", type=int, default=1650)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.concurrency < 1 or a.concurrency > 16 or a.requests < 1:
        p.error("concurrency 1..16; requests must be positive")
    if a.warmup_requests < 0 or not 1 <= a.prefix_repetitions <= 60:
        p.error("warmup nonnegative, repetitions 1..60")
    if a.min_prompt_tokens < 1 or a.min_prompt_tokens > a.max_prompt_tokens:
        p.error("min prompt tokens must be positive and <= max")
    if not 1 <= a.max_new_tokens <= 128 or a.timeout <= 0:
        p.error("output 1..128, timeout positive")
    return a


def verify_request_lengths(
    records: list[dict[str, Any]], *, lower: int, upper: int
) -> tuple[bool, dict[str, Any]]:
    ok = [r for r in records if r["status"] == "ok"]
    counts = [r.get("prompt_tokens") for r in ok]
    complete = bool(ok) and all(
        isinstance(n, int) and not isinstance(n, bool)
        for n in counts
    )
    if not complete:
        return False, {"prompt_usage_complete": False, "prompt_tokens": counts}
    within_bounds = all(lower <= n <= upper for n in counts)
    return within_bounds, {
        "prompt_usage_complete": True,
        "all_lengths_within_bounds": within_bounds,
        "minimum_actual_prompt_tokens": min(counts),
        "maximum_actual_prompt_tokens": max(counts),
        "mean_actual_prompt_tokens": sum(counts) / len(counts),
    }


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        endpoint = _endpoint(a.base_url)
    except ValueError as exc:
        print(f"BLOCKED: endpoint invalid: {exc}", file=sys.stderr)
        return 2

    # Same template and seed for both cache ON/OFF conditions.
    # The distinct-prefix seed is changed between separate trials.
    def send(index: int):
        return _request(
            index, endpoint=endpoint, model=a.model,
            # DISTINCT requests must not repeat across concurrency cases:
            # previous arms reused the same (seed,index) combination and
            # contaminated the later "no-reuse" negative control.
            prompt=make_prompt(
                index,
                seed=a.seed + (a.concurrency * 1009 if a.mode == "distinct" else 0),
                repetitions=a.prefix_repetitions,
                mode=a.mode,
            ),
            max_tokens=a.max_new_tokens, timeout=a.timeout, seed=a.seed,
            response_instruction=RESPONSE_INSTRUCTION,
        )

    warmups = [send(-i - 1) for i in range(a.warmup_requests)]
    if any(x["status"] != "ok" for x in warmups):
        print("BLOCKED: warmup request failed", file=sys.stderr)
        return 2
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        pending = [pool.submit(send, i) for i in range(a.requests)]
        records = sorted(
            (f.result() for f in as_completed(pending)),
            key=lambda r: r["index"],
        )
    wall = time.perf_counter() - started
    summary = summarize(records, wall)
    lengths_ok, length_check = verify_request_lengths(
        records, lower=a.min_prompt_tokens, upper=a.max_prompt_tokens
    )
    success = summary["succeeded"] == a.requests and lengths_ok
    result = {
        "status": "measured" if success else "invalid_or_blocked",
        "kind": "real_sse_prefix_cache_workload",
        "engine_assumption": "vllm",
        "declared_server_prefix_cache_state": a.declared_cache_state,
        "config": {
            "base_url": a.base_url, "model": a.model, "mode": a.mode,
            "concurrency": a.concurrency, "requests": a.requests,
            "warmup_requests": a.warmup_requests,
            "prefix_repetitions": a.prefix_repetitions,
            "min_prompt_tokens": a.min_prompt_tokens,
            "max_prompt_tokens": a.max_prompt_tokens,
            "max_new_tokens": a.max_new_tokens,
            "seed": a.seed,
            "distinct_prefix_case_seed": a.seed + a.concurrency * 1009,
            "distinct_prefix_ids_isolated_by_concurrency": True,
            "template_sha256": hashlib.sha256(
                (TECHNICAL_PARAGRAPH + RESPONSE_INSTRUCTION).encode("utf-8")
            ).hexdigest(),
            "prompt_text_saved": False,
        },
        "environment": environment(),
        "summary": summary,
        "prompt_length_gate": length_check,
        "requests": records,
    }
    write_json_atomic(a.output, result)
    print(
        f"Prefix caching arm: state={a.declared_cache_state}, mode={a.mode}, "
        f"concurrency={a.concurrency}, completed={summary['succeeded']}/{a.requests}, "
        f"prompt_tokens={summary['mean_prompt_tokens']}, "
        f"p95_TTFT_ms={summary['p95_ttft_ms']}, "
        f"output_tokens_s={summary['output_tokens_per_second']}; "
        f"validated={success}"
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
