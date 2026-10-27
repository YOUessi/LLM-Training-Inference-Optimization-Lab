"""Derive a conservative TTFT SLO versus throughput frontier from real runs.

This is a post-hoc analysis of the Stage 5 mixed-context Qwen2.5 serving
experiment, not a new GPU measurement. Two-seed min-throughput / max-TTFT
bounds are descriptive worst-of-two observations, not confidence intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_opt_lab.provenance import write_json_atomic

CONCURRENCIES=(1,2,4,8)
SEEDS=(42,137)
SLO_THRESHOLDS_MS=(100.,150.,250.,500.)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def analyze(evidence_dir: Path, slo_thresholds: tuple[float,...]=SLO_THRESHOLDS_MS) -> dict:
    runs=[]
    for c in CONCURRENCIES:
        for seed in SEEDS:
            path=evidence_dir/f"mixed_c{c}_seed{seed}.json"
            row=json.loads(path.read_text(encoding="utf-8"))
            if row["status"]!="measured" or row["kind"]!="vllm_mixed_context_real_generation_workload":
                raise ValueError(f"not a real measured vLLM mixed run: {path}")
            cfg=row["config"];summ=row["summary"]
            if cfg["concurrency"]!=c or cfg["seed"]!=seed:
                raise ValueError("mismatched parameters")
            if cfg["repeats_per_length"]!=6 or cfg["max_new_tokens"]!=32:
                raise ValueError("different workload configuration")
            if summ["succeeded"]!=24 or summ["failed"]!=0:
                raise ValueError("incomplete real requests")
            if summ["usage_coverage"]!=1. or summ["prompt_usage_coverage"]!=1.:
                raise ValueError("incomplete token usage")
            buckets=row["by_prompt_length"]
            if set(buckets)!=set(("short","medium","long","very_long")) or any(x["status"]!="pass" for x in buckets.values()):
                raise ValueError("invalid real prompt lengths")
            if summ["p95_ttft_ms"] is None or summ["output_tokens_per_second"] is None:
                raise ValueError("missing TTFT or throughput")
            runs.append({
                "concurrency":c,"seed":seed,
                "p95_ttft_ms":float(summ["p95_ttft_ms"]),
                "output_tokens_per_second":float(summ["output_tokens_per_second"]),
                "input_tokens_mean":float(summ["mean_prompt_tokens"]),
            })
    grouped=[]
    for c in CONCURRENCIES:
        xs=[x for x in runs if x["concurrency"]==c]
        assert len(xs)==2
        grouped.append({
            "concurrency":c,
            "seed_count":2,
            "worst_p95_ttft_ms":max(x["p95_ttft_ms"] for x in xs),
            "best_p95_ttft_ms":min(x["p95_ttft_ms"] for x in xs),
            "conservative_output_tps":min(x["output_tokens_per_second"] for x in xs),
            "highest_output_tps":max(x["output_tokens_per_second"] for x in xs),
        })
    decisions=[]
    for slo in slo_thresholds:
        if slo<=0:raise ValueError("SLO must be positive")
        eligible=[x for x in grouped if x["worst_p95_ttft_ms"]<=slo]
        choice=max(eligible,key=lambda x:x["conservative_output_tps"]) if eligible else None
        decisions.append({
            "p95_ttft_slo_ms":slo,
            "eligible_concurrencies":[x["concurrency"] for x in eligible],
            "selected_concurrency":choice["concurrency"] if choice else None,
            "selected_conservative_output_tps":choice["conservative_output_tps"] if choice else None,
            "actual_worst_p95_ttft_ms":choice["worst_p95_ttft_ms"] if choice else None,
        })
    return {
        "status":"derived_from_measured_evidence",
        "kind":"empirical_two_seed_ttft_slo_frontier",
        "not_new_gpu_experiment":True,
        "engine":"vllm",
        "gpu_source":"Tang RTX4090 Laptop",
        "source":"Stage5 cached-off mixed-context 4 lengths, 2 seeds, 24 SSE requests each concurrency",
        "caveats":[
            "Worst of two seeds is not a statistical confidence bound",
            "The SLO decision only applies to this pinned synthetic workload",
            "TTFT includes server request queueing and network latency",
            "No extrapolation beyond concurrency 1/2/4/8 or to other models",
        ],
        "runs":runs,
        "frontier_points":grouped,
        "slo_decisions":decisions,
    }


def main(argv: list[str] | None = None) -> int:
    a=parse_args(argv)
    report=analyze(a.evidence_dir)
    write_json_atomic(a.out,report)
    for x in report["slo_decisions"]:
        print(
            f"TTFT SLO {x['p95_ttft_slo_ms']:.0f}ms: "
            f"selected concurrency={x['selected_concurrency']}, "
            f"conservative throughput={x['selected_conservative_output_tps']}",
        )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
