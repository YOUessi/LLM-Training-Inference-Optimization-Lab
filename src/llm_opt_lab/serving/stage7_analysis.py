"""Analyze full 384-request Stage7 heterogeneous prose test.

Strictly validates 16 real SSE result files and the preregistered workload
contract before producing SLO/throughput comparison. No synthetic metrics.
Two seeded repetitions are descriptive, not a confidence bound.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from llm_opt_lab.provenance import write_json_atomic
from llm_opt_lab.serving.stage7_natural_bench import BUCKETS, GENERATION

SEEDS=(42,137)
CONCURRENCIES=(1,2,4,8)
MODES=("short","long")
SLOS=(100.,150.,250.,400.,600.,1000.)


def analyze(root: Path) -> dict:
    rows=[]
    commits=set()
    corpora=set()
    matched={}
    for seed in SEEDS:
        for mode in MODES:
            for c in CONCURRENCIES:
                path=root/f"{mode}_c{c}_seed{seed}.json"
                r=json.loads(path.read_text(encoding="utf-8"))
                if r.get("status")!="measured" or r.get("kind")!="stage7_natural_public_domain_vllm_mixed_context":
                    raise ValueError(f"not a real measured Stage7 output: {path}")
                cfg=r["config"]
                if (cfg["seed"],cfg["mode"],cfg["concurrency"])!=(seed,mode,c):
                    raise ValueError("case metadata mismatch")
                if cfg["max_new_tokens"]!=GENERATION[mode]["max_new_tokens"]:
                    raise ValueError("generation budget changed")
                if r["summary"]["succeeded"]!=24 or r["summary"]["failed"]!=0:
                    raise ValueError("a request failed")
                if r["summary"]["prompt_usage_coverage"]!=1.0 or r["summary"]["usage_coverage"]!=1.0:
                    raise ValueError("token usage incomplete")
                if r["generated_tokens_mean"]<GENERATION[mode]["min_mean_generated_tokens"]:
                    raise ValueError("actual generation too short for mode")
                if len(r["requests"])!=24:
                    raise ValueError("per-request source evidence missing")
                if set(r["bucket_statistics"])!=set(BUCKETS):
                    raise ValueError("mixed-length buckets missing")
                if any(x["status"]!="pass" or x["sent"]!=6
                       for x in r["bucket_statistics"].values()):
                    raise ValueError("input length gate incomplete")
                genres={item["genre"] for item in r["requests"]}
                if len(genres)!=5 or len({item["doc_id"] for item in r["requests"]})!=24:
                    raise ValueError("passage/genre assignment invalid")
                request_set=tuple(sorted((x["doc_id"],x["bucket"],x["prompt_sha256"]) for x in r["requests"]))
                key=(seed,mode)
                if key in matched and matched[key]!=request_set:
                    raise ValueError("concurrency arms were not fed identical documents and bucket prompts")
                matched[key]=request_set
                commits.add(r["environment"]["git_commit"])
                corpora.add(cfg["corpus_sha256"])
                rows.append({
                    "seed":seed, "mode":mode, "concurrency":c,
                    "p95_ttft_ms":r["summary"]["p95_ttft_ms"],
                    "p50_ttft_ms":r["summary"]["p50_ttft_ms"],
                    "p95_latency_ms":r["summary"]["p95_latency_ms"],
                    "output_tokens_per_second":r["summary"]["output_tokens_per_second"],
                    "requests_per_second":r["summary"]["requests_per_second"],
                    "mean_real_generated_tokens":r["generated_tokens_mean"],
                    "actual_prompt_by_bucket":{
                        b:{
                          "min":r["bucket_statistics"][b]["min_actual_prompt_tokens"],
                          "max":r["bucket_statistics"][b]["max_actual_prompt_tokens"],
                          "p95_ttft_ms":r["bucket_statistics"][b]["p95_ttft_ms"],
                        }
                        for b in BUCKETS
                    },
                    "raw_file":path.name,
                })
    if len(commits)!=1 or len(corpora)!=1:
        raise ValueError("multiple source commits or corpora in one experiment")
    for seed in SEEDS:
        if matched[(seed,"short")]!=matched[(seed,"long")]:
            raise ValueError("same seed short/long used different frozen input passages")
    totals={}
    frontier={}
    for mode in MODES:
        pts=[]
        for c in CONCURRENCIES:
            rep=[x for x in rows if x["mode"]==mode and x["concurrency"]==c]
            if len(rep)!=2:
                raise ValueError("two-seed estimate incomplete")
            pts.append({
                "concurrency":c,
                "observations":rep,
                "p95_ttft_best_ms":min(x["p95_ttft_ms"] for x in rep),
                "p95_ttft_worst_ms":max(x["p95_ttft_ms"] for x in rep),
                "out_tps_min":min(x["output_tokens_per_second"] for x in rep),
                "out_tps_max":max(x["output_tokens_per_second"] for x in rep),
                "real_generated_tokens_mean":statistics.mean(x["mean_real_generated_tokens"] for x in rep),
            })
        frontier[mode]=pts
        totals[mode]=[{
            "p95_ttft_slo_ms":slo,
            "eligible_concurrencies":[p["concurrency"] for p in pts if p["p95_ttft_worst_ms"]<=slo],
            "selected_concurrency":max(
                (p for p in pts if p["p95_ttft_worst_ms"]<=slo),
                key=lambda x:x["out_tps_min"],default=None
            )["concurrency"] if any(p["p95_ttft_worst_ms"]<=slo for p in pts) else None,
            "conservative_out_tps":max(
                (p["out_tps_min"] for p in pts if p["p95_ttft_worst_ms"]<=slo),
                default=None
            ),
        } for slo in SLOS]
    return {
        "status":"pass",
        "kind":"stage7_real_varied_prose_prefill_decode_slo_analysis",
        "not_new_gpu_measurement":True,
        "experiment_source_commit":next(iter(commits)),
        "corpus_jsonl_sha256":next(iter(corpora)),
        "total_measured_requests":len(rows)*24,
        "unique_documents_per_case":24,
        "source_works":5,
        "case_count":len(rows),
        "cases":rows,
        "two_seed_observed_frontier":frontier,
        "observed_slo_decisions":totals,
        "limitations":[
            "One GPU, one Qwen2.5-1.5B model and frozen public-domain prose genres",
            "Two-seed observed extrema are not a confidence interval",
            "Short vs long comparisons change output instructions and token budget",
            "No causal attribution to KV cache or scheduling internals",
            "P95 on just 24 queries/case is exploratory, not an SLA guarantee",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        d=analyze(a.raw_dir)
        write_json_atomic(a.output,d)
        print("STAGE7 ANALYSIS PASS",d["total_measured_requests"],
              "requests from",d["case_count"],"real GPU cases")
        for mode, pts in d["two_seed_observed_frontier"].items():
            print("MODE",mode)
            for q in pts:
                print("  concurrency",q["concurrency"],"TTFT",
                      round(q["p95_ttft_best_ms"],1),round(q["p95_ttft_worst_ms"],1),
                      "output_tps",round(q["out_tps_min"],1),round(q["out_tps_max"],1),
                      "generated mean",round(q["real_generated_tokens_mean"],1))
        return 0
    except (OSError,KeyError,ValueError,TypeError) as exc:
        print(f"INVALID STAGE7 RESULT: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
