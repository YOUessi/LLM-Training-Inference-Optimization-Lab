"""Strict 288-request, multilingual open-loop SSE experiment analysis.

Every statistic derives from actual archived per-request SSE timings; no
fake GPU values and no general statistical P95 guarantee. The analysis
never reclassifies a protocol-invalid run as measured.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from llm_opt_lab.provenance import write_json_atomic
from llm_opt_lab.serving.bench_client import quantile
from llm_opt_lab.serving.stage8_arrivals import MODES, RATES, SEEDS, SLO_MS

SOURCE_SHA = "8a6053a1d4ee8fa3850dd362a9ebb6ab9039b7f8"
CORPUS_SHA = "c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c"


def analyze(root: Path) -> dict:
    rows = []
    comparisons: dict[int, set[tuple]] = {}
    for seed in SEEDS:
        for mode in MODES:
            for rate in RATES:
                file=root / f"{mode}_rate{int(rate)}_seed{seed}.json"
                p=json.loads(file.read_text(encoding="utf-8"))
                if p.get("status") != "measured" or p.get("kind") != "real_vllm_stage8_multilingual_openloop":
                    raise ValueError(f"not an accepted real Stage8 SSE experiment: {file}")
                cfg, s = p["config"], p["summary"]
                if cfg["seed"]!=seed or cfg["mode"]!=mode or cfg["rate_per_second"]!=rate:
                    raise ValueError("frozen case dimensions mismatch")
                if cfg["frozen_request_sha256"]!=CORPUS_SHA or p["environment"]["git_commit"]!=SOURCE_SHA:
                    raise ValueError("source revision or frozen request corpus not verified")
                if cfg["output_cap_tokens"]!=128 or cfg["requests"]!=24:
                    raise ValueError("generation cap or request count changed")
                if s["succeeded"]!=24 or s["failed"]!=0:
                    raise ValueError("not all real SSE requests successful")
                if s["prompt_usage_coverage"]!=1. or s["usage_coverage"]!=1.:
                    raise ValueError("actual input/output usage missing")
                if not p["all_actual_input_tokens_in_bounds"] or p["length_gates"]["status"]!="pass":
                    raise ValueError("input or natural stop-length acceptance gate failed")
                requests=p["requests"]
                if len(requests)!=24:
                    raise ValueError("per-request source evidence incomplete")
                langs={q["language"] for q in requests}
                if langs != {"en","zh-cn","ja"}:
                    raise ValueError("one language cohort missing")
                for language in langs:
                    if sum(q["language"]==language for q in requests)!=8:
                        raise ValueError("language counts unbalanced")
                if len({q["request_id"] for q in requests})!=24:
                    raise ValueError("request ids reused")
                if sum(q["bucket"]=="short" for q in requests)!=6:
                    raise ValueError("real input buckets unbalanced")
                if any(q["status"]!="ok" or q.get("prompt_tokens") is None
                       or abs(q["prompt_tokens"]-q["target_prompt_tokens"])>95
                       or q.get("completion_tokens") is None for q in requests):
                    raise ValueError("provider token gates violated")
                if any(q["client_executor_wait_ms"]<0 or q["ttft_from_due_ms"] is None
                       for q in requests):
                    raise ValueError("scheduled arrival / client queue trace missing")
                trace=tuple((q["request_id"],q["bucket"],q["prompt_sha256"]) for q in requests)
                if seed in comparisons and trace!=next(iter(comparisons[seed])):
                    raise ValueError("same seed different workload order/content across modes/rates")
                comparisons.setdefault(seed,set()).add(trace)
                full_ttft=[q["ttft_from_due_ms"] for q in requests]
                if abs(quantile(full_ttft,.95)-p["p95_ttft_from_scheduled_arrival_ms"])>.001:
                    raise ValueError("arrival-to-TTFT quantile calculation mismatch")
                if s["p95_ttft_ms"] is None or s["output_tokens_per_second"] is None:
                    raise ValueError("server TTFT or output throughput unavailable")
                if any(p["ttft_slo_violations"][str(int(t))] !=
                       sum(q["ttft_from_due_ms"]>t for q in requests) for t in SLO_MS):
                    raise ValueError("SLO violations mismatched actual per-request times")
                rows.append({
                    "seed":seed,"mode":mode,"target_rate":rate,
                    "p95_arrival_to_ttft_ms":p["p95_ttft_from_scheduled_arrival_ms"],
                    "p95_client_send_ttft_ms":s["p95_ttft_ms"],
                    "p95_client_executor_wait_ms":p["p95_executor_queue_wait_ms"],
                    "p95_request_latency_ms":s["p95_latency_ms"],
                    "output_tokens_per_second":s["output_tokens_per_second"],
                    "completed_requests_per_second":p["completed_requests_per_second"],
                    "empirical_offered_rate":p["offered_arrival_rate_empirical_per_second"],
                    "mean_actual_output_tokens":p["length_gates"]["realized_mean"],
                    "real_output_stopped_early_fraction":p["length_gates"]["fraction_stopped_before_cap"],
                    "actual_input_range_tokens":[p["actual_input_token_min"],p["actual_input_token_max"]],
                    "kv_cache_fraction_peak":p["gpu_kv_cache_gauge"]["peak_fraction"],
                    "kv_cache_samples":p["gpu_kv_cache_gauge"]["samples"],
                    "slo_violations":p["ttft_slo_violations"],
                    "raw_file":file.name,
                })
    if len(rows)!=12:
        raise ValueError("missing 12 factorial cases")
    grouped=[]
    for mode in MODES:
        for rate in RATES:
            two=[x for x in rows if x["mode"]==mode and x["target_rate"]==rate]
            if {x["seed"] for x in two}!={42,137}:
                raise ValueError("two-seed factorial cell incomplete")
            grouped.append({
                "mode":mode,"target_rate":rate,
                "worst_p95_arrival_ttft_ms":max(x["p95_arrival_to_ttft_ms"] for x in two),
                "best_p95_arrival_ttft_ms":min(x["p95_arrival_to_ttft_ms"] for x in two),
                "worst_p95_client_send_ttft_ms":max(x["p95_client_send_ttft_ms"] for x in two),
                "conservative_output_tokens_per_second":min(x["output_tokens_per_second"] for x in two),
                "max_output_tokens_per_second":max(x["output_tokens_per_second"] for x in two),
                "max_p95_executor_wait_ms":max(x["p95_client_executor_wait_ms"] for x in two),
                "mean_real_generated_tokens":statistics.mean(x["mean_actual_output_tokens"] for x in two),
                "slo_250ms_violation_counts":[x["slo_violations"]["250"] for x in two],
            })
    admission=[]
    for mode in MODES:
        cell=[x for x in grouped if x["mode"]==mode]
        for threshold in SLO_MS:
            candidates=[x for x in cell if x["worst_p95_arrival_ttft_ms"]<=threshold]
            best=max(candidates,key=lambda x:x["conservative_output_tokens_per_second"],
                     default=None)
            admission.append({
                "mode":mode,"p95_arrival_ttft_limit_ms":threshold,
                "qualifying_target_rates":[x["target_rate"] for x in candidates],
                "selected_target_rate":best["target_rate"] if best else None,
                "conservative_output_tokens_per_second": (
                    best["conservative_output_tokens_per_second"] if best else None
                ),
            })
    return {
        "status":"pass",
        "kind":"stage8_real_multilingual_openloop_natural_length_analysis",
        "source_commit":SOURCE_SHA,
        "frozen_requests_sha256":CORPUS_SHA,
        "measured_requests":12*24,
        "case_count":len(rows),
        "not_a_new_gpu_experiment":True,
        "per_case":rows,
        "descriptive_two_seed_cells":grouped,
        "experimental_slo_admission_decisions":admission,
        "limitations":[
            "Two seeds and only 24 arrivals/case do not provide P95 statistical confidence",
            "Language classes are all from cloud-native Kubernetes documentation",
            "Arrival-to-TTFT includes client executor waiting and server processing",
            "Different instructions change output length and prompt length; not an isolated scheduler ablation",
            "The open-loop offered rate is a sampled exponential arrival process; use empirical and target rates separately",
            "A fast label response can be client-network/stream overhead bound even when token/s is low",
            "KV gauge sampling is real Prometheus, unavailable is null, not zero",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        result=analyze(a.raw_dir)
        write_json_atomic(a.out,result)
        print("STAGE8_ANALYSIS_PASS",result["measured_requests"],"12 case cells")
        for x in result["descriptive_two_seed_cells"]:
            print(x["mode"],"target rate",x["target_rate"],
                  "worst_p95_arrival_ttft",round(x["worst_p95_arrival_ttft_ms"],1),
                  "minimum_output_tok_s",round(x["conservative_output_tokens_per_second"],1),
                  "max_executor_queue_p95",round(x["max_p95_executor_wait_ms"],1))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f"INVALID STAGE8 ANALYSIS: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
