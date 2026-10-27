"""Analyze a complete, real ON/OFF x shared/distinct prefix experiment.

Reads only raw JSON written by our real SSE benchmark and the Prometheus
cache-token counter gates. Fails closed for missing cases, contaminated
negative controls, failed HTTP requests or unmatched prompt lengths.

Outputs a single machine-readable report with all paired experiment rows.
Two seeds are not enough for statistical-confidence or universal causality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from llm_opt_lab.provenance import write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def summarize_pair(off: dict, on: dict) -> dict:
    """Match one exact (mode,seed,concurrency) condition."""
    for key in ("mode", "seed", "concurrency"):
        if off[key] != on[key]:
            raise ValueError(f"cache on/off not matched for {key}")
    if abs(off["mean_prompt_tokens"] - on["mean_prompt_tokens"]) > 2:
        raise ValueError("cache ON/OFF mismatch in actual input token counts")
    if off["p95_ttft_ms"] <= 0 or on["p95_ttft_ms"] <= 0:
        raise ValueError("TTFT must be measured and strictly positive")
    if off["output_tokens_per_second"] <= 0 or on["output_tokens_per_second"] <= 0:
        raise ValueError("output throughput must be measured and positive")
    return {
        "mode": off["mode"], "seed": off["seed"],
        "concurrency": off["concurrency"],
        "mean_prompt_tokens": (off["mean_prompt_tokens"] + on["mean_prompt_tokens"]) / 2.,
        "off_p95_ttft_ms": off["p95_ttft_ms"],
        "on_p95_ttft_ms": on["p95_ttft_ms"],
        "ttft_off_over_on": off["p95_ttft_ms"] / on["p95_ttft_ms"],
        "off_output_tokens_per_second": off["output_tokens_per_second"],
        "on_output_tokens_per_second": on["output_tokens_per_second"],
        "throughput_on_over_off": on["output_tokens_per_second"] / off["output_tokens_per_second"],
        "off_metric": off["metric"],
        "on_metric": on["metric"],
    }


def aggregate_two_seeds(pairs: list[dict]) -> list[dict]:
    """Descriptive means/ranges, explicitly not an inferential confidence interval."""
    result = []
    for mode in ("shared", "distinct"):
        for concurrency in (1, 8):
            x = [a for a in pairs if a["mode"] == mode and a["concurrency"] == concurrency]
            if len(x) != 2 or {a["seed"] for a in x} != {42, 137}:
                raise ValueError("expected two unique seed replicates")
            ratios = [r["ttft_off_over_on"] for r in x]
            throughputs = [r["throughput_on_over_off"] for r in x]
            result.append({
                "mode": mode,
                "concurrency": concurrency,
                "seed_count": len(x),
                "mean_ttft_off_over_on": statistics.mean(ratios),
                "min_ttft_off_over_on": min(ratios),
                "max_ttft_off_over_on": max(ratios),
                "mean_throughput_on_over_off": statistics.mean(throughputs),
                "min_throughput_on_over_off": min(throughputs),
                "max_throughput_on_over_off": max(throughputs),
                "mean_prefix_cache_hit_fraction_on": statistics.mean(
                    r["on_metric"]["hit_fraction"] for r in x
                ),
            })
    return result


def analyze(root: Path) -> dict:
    runs=[]
    commit_set=set()
    source_files=[]
    for state in ("off","on"):
        log=(root/state/"server.log").read_text(encoding="utf-8",errors="replace")
        expected="True" if state=="on" else "False"
        if ("enable_prefix_caching': "+expected not in log and
            "enable_prefix_caching="+expected not in log):
            raise ValueError(f"server log did not confirm actual cache state={state}")
        for seed in (42,137):
            for mode in ("shared","distinct"):
                for c in (1,8):
                    label=f"{mode}-c{c}-seed{seed}"
                    path=root/state/(label+".json")
                    metric_path=root/state/(label+"-metrics.json")
                    raw=path.read_bytes()
                    r=json.loads(raw)
                    metric=json.loads(metric_path.read_text())
                    if r["status"]!="measured" or metric["status"]!="pass":
                        raise ValueError(f"unacceptable experiment status: {path}")
                    if r["declared_server_prefix_cache_state"]!=state:
                        raise ValueError("server cache state mismatch in result")
                    a=r["config"]
                    if (a["mode"],a["concurrency"],a["seed"])!=(mode,c,seed):
                        raise ValueError("workload config changed mid-ablation")
                    if not a.get("distinct_prefix_ids_isolated_by_concurrency"):
                        raise ValueError("cross-case distinct prefix isolation missing")
                    s=r["summary"]
                    if s["succeeded"] != 24 or s["failed"]!=0:
                        raise ValueError("incomplete request count")
                    if s["prompt_usage_coverage"] != 1.0 or s["usage_coverage"] != 1.0:
                        raise ValueError("incomplete token usage")
                    if not r["prompt_length_gate"]["all_lengths_within_bounds"]:
                        raise ValueError("invalid actual input length")
                    if not 900 <= s["mean_prompt_tokens"] <= 1450:
                        raise ValueError("measured prompt token length out of bounds")
                    if a["max_new_tokens"] != 32 or a["prefix_repetitions"]!=35:
                        raise ValueError("model generation budget or source prompt changed")
                    commit_set.add(r["environment"]["git_commit"])
                    source_files.append({
                        "path":str(path.relative_to(root)),
                        "sha256":hashlib.sha256(raw).hexdigest(),
                        "cache_metric_path":str(metric_path.relative_to(root)),
                    })
                    runs.append({
                        "cache_state":state, "seed":seed,"mode":mode,
                        "concurrency":c,"mean_prompt_tokens":s["mean_prompt_tokens"],
                        "p95_ttft_ms":s["p95_ttft_ms"],
                        "p50_ttft_ms":s["p50_ttft_ms"],
                        "output_tokens_per_second":s["output_tokens_per_second"],
                        "p95_latency_ms":s["p95_latency_ms"],
                        "successes":s["succeeded"], "metric":metric,
                        "source_file":str(path.relative_to(root)),
                    })
    if len(commit_set)!=1:
        raise ValueError("experiment ran at multiple source commits")
    expected_cases={(seed,mode,c) for seed in (42,137) for mode in ("shared","distinct") for c in (1,8)}
    pairs=[]
    for seed,mode,c in sorted(expected_cases):
        a=[x for x in runs if x["seed"]==seed and x["mode"]==mode and x["concurrency"]==c]
        if len(a)!=2 or {r["cache_state"] for r in a}!={"on","off"}:
            raise ValueError("cache ON/OFF pair incomplete")
        pairs.append(summarize_pair(
            next(r for r in a if r["cache_state"]=="off"),
            next(r for r in a if r["cache_state"]=="on"),
        ))
    return {
        "status":"pass",
        "kind":"vllm_prefix_cache_on_off_factored_ablation",
        "source_git_commit":next(iter(commit_set)),
        "model":"Qwen2.5-1.5B-Instruct",
        "engine":"vllm-0.19.1",
        "not_a_production_benchmark":True,
        "n_independent_seeds":2,
        "measured_real_requests":sum(x["successes"] for x in runs),
        "arms":runs,
        "matched_on_off_pairs":pairs,
        "descriptive_aggregate":aggregate_two_seeds(pairs),
        "file_integrity_manifest":source_files,
        "limitations":[
            "Single RTX4090 Laptop GPU and fixed synthetic repetitive context",
            "Two seeds and two orderings: ranges are descriptive, not confidence intervals",
            "TTFT is client first nonempty SSE content fragment, not raw GPU prefill kernel time",
            "Paired comparisons are only within matching prompt mode, seed and concurrency",
            "Prefix cache metrics count cached tokens; cache OFF may report zero queries",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    a=parse_args(argv)
    try:
        report=analyze(a.raw_dir)
        write_json_atomic(a.output,report)
        print("PREFIX CACHE ANALYSIS PASSED: requests",report["measured_real_requests"],
              "pairs",len(report["matched_on_off_pairs"]))
        for row in report["descriptive_aggregate"]:
            print(row["mode"],"concurrency",row["concurrency"],
                  "TTFT off/on",round(row["mean_ttft_off_over_on"],2),
                  "throughput on/off",round(row["mean_throughput_on_over_off"],2),
                  "ON hit fraction",round(row["mean_prefix_cache_hit_fraction_on"],3))
        return 0
    except (OSError,KeyError,ValueError,TypeError) as exc:
        print(f"INVALID EXPERIMENT: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
