"""Real vLLM automatic prefix-cache retention and eviction probe.

Use a bounded GPU block override on a dedicated server, no shared GPU tasks.
Verify behavior from server prefix_cache_queries_total / hits_total counters,
not TTFT changes alone. A failed eviction is preserved, not labeled success.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic
from llm_opt_lab.serving.bench_client import _endpoint, _request
from llm_opt_lab.serving.prefix_cache_bench import make_prompt
from llm_opt_lab.serving.prefix_cache_metrics import METRICS, parse_totals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8026")
    p.add_argument("--model", required=True)
    p.add_argument("--distinct-fillers", type=int, default=12)
    p.add_argument("--prefix-repetitions", type=int, default=35)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=830)
    p.add_argument("--num-gpu-blocks-override", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if not 4 <= a.distinct_fillers <= 80:
        p.error("distinct-fillers must be 4..80")
    if not 8 <= a.prefix_repetitions <= 50:
        p.error("prefix-repetitions must be 8..50")
    if not 1 <= a.max_new_tokens <= 64:
        p.error("max-new-tokens must be 1..64")
    if not 32 <= a.num_gpu_blocks_override <= 4096:
        p.error("num-gpu-blocks-override must be 32..4096")
    return a


def ratio_between(
    before: dict[str, float | None],
    after: dict[str, float | None],
) -> dict[str, float]:
    """Strict counters: no missing series, resets, or impossible hit fractions."""
    names = METRICS
    for key in names:
        if before.get(key) is None or after.get(key) is None:
            raise ValueError(f"required vLLM cache counter unavailable: {key}")
    queries = after[names[0]] - before[names[0]]
    hits = after[names[1]] - before[names[1]]
    if queries <= 0 or hits < 0 or hits > queries + 1e-8:
        raise ValueError(f"invalid prefix counter deltas queries={queries}, hits={hits}")
    return {
        "queried_tokens": queries,
        "cache_hit_tokens": hits,
        "cache_hit_fraction": hits / queries,
    }


def fetch_counters(base_url: str) -> dict[str, float | None]:
    endpoint = base_url.rstrip("/") + "/metrics"
    with urllib.request.urlopen(endpoint, timeout=12) as response:
        if response.status != 200:
            raise RuntimeError(f"metrics HTTP {response.status}")
        return parse_totals(response.read().decode("utf-8"))


def evaluate(probes: list[dict]) -> dict[str, bool]:
    """Intentionally strict, evidence-backed probe gates."""
    stages = {p["stage"]: p for p in probes if p["stage"] in
              ("initial_miss", "immediate_replay", "post_pressure_replay", "restored_replay")}
    if len(stages) != 4:
        raise ValueError("missing mandatory stages")
    return {
        "initial_cold_request_below_10pct": stages["initial_miss"]["cache"]["cache_hit_fraction"] <= .10,
        "second_identical_request_above_75pct": stages["immediate_replay"]["cache"]["cache_hit_fraction"] >= .75,
        "replay_after_pressure_below_25pct": stages["post_pressure_replay"]["cache"]["cache_hit_fraction"] <= .25,
        "replay_after_warming_above_75pct": stages["restored_replay"]["cache"]["cache_hit_fraction"] >= .75,
    }


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        _endpoint(a.base_url)
        events = []
        seed = a.seed
        anchor = make_prompt(0, seed=seed, repetitions=a.prefix_repetitions, mode="shared")

        def one(stage: str, prompt: str, index: int) -> None:
            before = fetch_counters(a.base_url)
            row = _request(index, endpoint=_endpoint(a.base_url),
                           model=a.model, prompt=prompt,
                           max_tokens=a.max_new_tokens, timeout=90.0, seed=seed,
                           response_instruction="Give a one-sentence technical summary.")
            after = fetch_counters(a.base_url)
            if row["status"] != "ok":
                raise RuntimeError(f"{stage}: request failed {row.get('error_type')}")
            if not isinstance(row["prompt_tokens"], int) or not 250 <= row["prompt_tokens"] <= 1700:
                raise ValueError(f"{stage}: actual prompt tokens out of contract {row['prompt_tokens']}")
            metric = ratio_between(before, after)
            events.append({
                "stage": stage,
                "request_index": index,
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "ttft_ms": row["ttft_ms"],
                "latency_ms": row["latency_ms"],
                "cache": metric,
            })
            print(f"{stage}: cached {metric['cache_hit_fraction']:.1%}, "
                  f"TTFT {row['ttft_ms']:.1f}ms, prompt tokens {row['prompt_tokens']}",flush=True)

        # Exact same content + task index: a reproducible replay experiment.
        one("initial_miss", anchor, 0)
        one("immediate_replay", anchor, 0)
        for i in range(a.distinct_fillers):
            unique = make_prompt(100+i, seed=seed + 123,
                                 repetitions=a.prefix_repetitions, mode="distinct")
            one(f"unique_filler_{i:03d}", unique, 100+i)
        one("post_pressure_replay", anchor, 0)
        one("restored_replay", anchor, 0)
        gate = evaluate(events)
        result = {
            "status":"pass" if all(gate.values()) else "invalid_or_no_eviction_observed",
            "kind":"vllm_prefix_cache_capacity_eviction_probe",
            "engine":"vllm",
            "source":"actual streaming requests plus Prometheus vLLM prefix tokens",
            "environment":environment(),
            "config":{
                "model":a.model,
                "base_url":a.base_url,
                "num_gpu_blocks_override":a.num_gpu_blocks_override,
                "distinct_fillers":a.distinct_fillers,
                "prefix_repetitions":a.prefix_repetitions,
                "max_new_tokens":a.max_new_tokens,
                "seed":a.seed,
            },
            "gates":gate,"events":events,
            "caveat":"Cache eviction inference is supported by loss of prefix token hits, not a direct per-block eviction counter. Production representativeness is not claimed.",
        }
        write_json_atomic(a.output,result)
        print("PROBE_RESULT",result["status"],gate,flush=True)
        return 0 if all(gate.values()) else 1
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
