"""Verify actual admission-policy results against all-offered-request accounting.

A policy is NOT considered improved merely because accepted-only TTFT falls.
All offered requests, rejected/failed requests, and useful output throughput
remain explicit. This analyzer is CPU-only and recomputes from raw GPU logs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llm_opt_lab.provenance import write_json_atomic
from llm_opt_lab.serving.stage10_admission import accounting, offered_slo_compliance

SOURCE_SHA="1f654f7404a7cc7e1cd5a88f28fc0615349a154f"
FROZEN_SHA="c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c"
MODES=("label","detailed")
SEEDS=(42,137)
POLICIES=("unlimited","bounded")


def analyze(root: Path) -> dict:
    pairs=[]
    for mode in MODES:
        for seed in SEEDS:
            runs=[]
            for policy in POLICIES:
                path=root/f"{policy}_{mode}_seed{seed}.json"
                r=json.loads(path.read_text(encoding="utf-8"))
                c=r["config"]
                summary=r["summary"]
                if r.get("status")!="measured" or r.get("kind")!="stage10_real_client_admission_policy_study":
                    raise ValueError(f"not a valid actual controlled GPU run: {path}")
                if r["environment"]["git_commit"]!=SOURCE_SHA or c["frozen_requests_sha256"]!=FROZEN_SHA:
                    raise ValueError("source commit or frozen request identity changed")
                if (c["policy"],c["mode"],c["seed"])!=(policy,mode,seed):
                    raise ValueError("policy/mode/seed mismatch")
                if c["fixed_offered_rate"]!=8 or c["max_client_workers"]!=32 or c["output_cap_tokens"]!=128:
                    raise ValueError("benchmark controls changed")
                if c["schedule"]!="seeded_fixed_duration_exponential_spacings":
                    raise ValueError("wrong arrival time schedule")
                if c["bounded_inflight"]!=8 or c["bounded_queue_slots"]!=4 or c["admission_deadline_ms"]!=250:
                    raise ValueError("admission policy not actually frozen")
                offered=r["requests"]
                if len(offered)!=24:
                    raise ValueError("unaccounted offered requests")
                if summary!=accounting(offered,r["elapsed_s"]):
                    raise ValueError("machine summary not equal to raw offered-request accounting")
                if not r["server_queue_monitor"]["available"] or not r["gpu_kv_monitor"]["available"]:
                    raise ValueError("real server queue/KV instrumentation unavailable")
                if summary["errors"]!=0 or summary["completed"]+summary["rejected"]!=24:
                    raise ValueError("unexplained request error or missing status")
                if policy=="unlimited" and summary["rejected"]!=0:
                    raise ValueError("unlimited baseline unexpectedly rejected")
                if r["accepted_output_length_gate"]["status"]!="pass":
                    raise ValueError("model output length gate failed for completions")
                for language in ("en","zh-cn","ja"):
                    if summary["by_language"][language]["offered"]!=8:
                        raise ValueError("multilingual request distribution not balanced")
                inputs=[(x["request_id"],x["prompt_sha256"],x["scheduled_offset_s"])
                        for x in offered]
                if len(set(x[0] for x in inputs))!=24:
                    raise ValueError("duplicate requests")
                # All-offered denominator and request-level SLO recheck.
                for limit in (100.,250.,500.,1000.):
                    key=str(int(limit))
                    if summary["slo_all_offered"][key] != offered_slo_compliance(offered,limit):
                        raise ValueError("all-offered SLO was mislabeled")
                if summary["rejected"] and summary["p95_all_offered_ttft_ms"] is not None:
                    raise ValueError("all-offered P95 improperly omits rejected requests")
                runs.append({
                    "policy":policy,"mode":mode,"seed":seed,
                    "inputs":inputs,"schedule":c["schedule_offsets_s"],
                    "offered":24,
                    "completed":summary["completed"],
                    "rejected":summary["rejected"],
                    "errors":summary["errors"],
                    "all_offered_slo_250_fraction":
                        summary["slo_all_offered"]["250"]["compliance_fraction"],
                    "all_offered_slo_500_fraction":
                        summary["slo_all_offered"]["500"]["compliance_fraction"],
                    "useful_output_tokens_per_second":
                        summary["useful_output_tokens_per_second"],
                    "accepted_only_p95_ttft_ms":
                        summary["p95_completed_only_arrival_ttft_ms"],
                    "all_offered_p95_ttft_ms":
                        summary["p95_all_offered_ttft_ms"],
                    "p95_client_admission_wait_ms":
                        summary["p95_admission_queue_wait_ms"],
                    "server_peak_waiting_sampled":
                        r["server_queue_monitor"]["max_waiting"],
                    "rejection_reasons":
                        summary["rejection_reasons"],
                    "by_language":summary["by_language"],
                    "source_file":path.name,
                })
            before,after=runs
            if before["schedule"]!=after["schedule"] or before["inputs"]!=after["inputs"]:
                raise ValueError("offered arrival schedule/request identity differed between policies")
            pairs.append({
                "mode":mode,"seed":seed,
                "unlimited":{k:v for k,v in before.items() if k not in ("inputs","schedule")},
                "bounded":{k:v for k,v in after.items() if k not in ("inputs","schedule")},
                "delta_all_offered_250ms_slo_fraction":
                    after["all_offered_slo_250_fraction"]-
                    before["all_offered_slo_250_fraction"],
                "delta_useful_tokens_per_second":
                    after["useful_output_tokens_per_second"]-
                    before["useful_output_tokens_per_second"],
                "new_rejected_count":after["rejected"]-before["rejected"],
            })
    return {
        "status":"pass",
        "kind":"stage10_matched_real_client_admission_ablation",
        "source_git_sha":SOURCE_SHA,
        "frozen_request_sha":FROZEN_SHA,
        "total_offered_requests":len(pairs)*2*24,
        "paired_cases":pairs,
        "analysis_not_a_new_gpu_run":True,
        "no_claim_of_universal_improvement":True,
        "limitations":[
            "Client-side admission is not a vLLM scheduler code change",
            "Dropping offered requests may reduce accepted-only P95 while harming all-offered service",
            "Only 24 requests/condition, two seeds, one model and Kubernetes domain",
            "Server Prometheus queue gauge sampling misses short transient events",
            "Executor policy uses semaphores without strict FIFO fairness guarantee",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        result=analyze(a.raw_dir)
        write_json_atomic(a.output,result)
        print("STAGE10_ANALYSIS_PASS",result["total_offered_requests"],flush=True)
        for cell in result["paired_cases"]:
            print(cell["mode"],cell["seed"],
                  "bounded_rejections",cell["bounded"]["rejected"],
                  "all_offered_250ms_slo_change",
                  round(cell["delta_all_offered_250ms_slo_fraction"],3),
                  "useful_token_tps_delta",
                  round(cell["delta_useful_tokens_per_second"],1))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f"STAGE10_ANALYSIS_INVALID: {type(exc).__name__}: {exc}",flush=True)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
