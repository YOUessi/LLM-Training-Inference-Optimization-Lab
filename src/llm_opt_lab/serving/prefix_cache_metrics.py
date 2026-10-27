"""Check vLLM prefix cache real Prometheus counter deltas.

Do not mark a negative-control experiment valid simply because HTTP requests
succeeded. If "distinct" cases unexpectedly have a high cache hit fraction,
the experiment is contaminated, not an optimization success.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from llm_opt_lab.provenance import write_json_atomic


METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
)


def parse_totals(prometheus: str) -> dict[str, float | None]:
    results: dict[str, float | None] = {name: None for name in METRICS}
    for line in prometheus.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if name not in results:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            raise ValueError(f"malformed metric line: {name}")
        number = float(parts[-1])
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"invalid nonfinite/negative metric: {name}")
        results[name] = (results[name] or 0.) + number
    return results


def analyze(before: str, after: str, *, cache_state: str,
            mode: str) -> dict:
    if cache_state not in ("on", "off") or mode not in ("shared", "distinct"):
        raise ValueError("unsupported cache state/prefix mode")
    old, new = parse_totals(before), parse_totals(after)
    if any(old[k] is None or new[k] is None for k in METRICS):
        return {
            "status": "invalid_missing_counters",
            "cache_state": cache_state, "prefix_mode": mode,
            "before": old, "after": new,
        }
    query = new[METRICS[0]] - old[METRICS[0]]
    hits = new[METRICS[1]] - old[METRICS[1]]
    if query < 0 or hits < 0 or hits > query + 1e-9:
        status = "invalid_counter_reset_or_mismatch"
        frac = None
    elif cache_state == "off":
        status = "pass" if query == 0 and hits == 0 else "fail_caching_off_counters_nonzero"
        frac = None
    elif query == 0:
        status, frac = "fail_cache_on_queries_zero", None
    else:
        frac = hits / query
        if mode == "shared":
            status = "pass" if frac >= 0.70 else "fail_shared_prefix_not_reused"
        else:
            status = "pass" if frac <= 0.10 else "fail_distinct_prefix_reused"
    return {
        "status": status,
        "cache_state": cache_state,
        "prefix_mode": mode,
        "before": old, "after": new,
        "delta_queried_tokens": query,
        "delta_hit_tokens": hits,
        "hit_fraction": frac,
        "acceptance": {
            "off": "queried/hit counters must both remain zero",
            "on_shared": "hit fraction >= 0.70",
            "on_distinct": "hit fraction <= 0.10",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--cache-state", choices=("on", "off"), required=True)
    p.add_argument("--mode", choices=("shared", "distinct"), required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    result = analyze(
        a.before.read_text(), a.after.read_text(),
        cache_state=a.cache_state, mode=a.mode,
    )
    write_json_atomic(a.output, result)
    print(
        f"PREFIX_METRIC_GATE {a.cache_state}/{a.mode} status={result['status']} "
        f"query_delta={result.get('delta_queried_tokens')} "
        f"hit_delta={result.get('delta_hit_tokens')} "
        f"hit_fraction={result.get('hit_fraction')}",
        flush=True,
    )
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
