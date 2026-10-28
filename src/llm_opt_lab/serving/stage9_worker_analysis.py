"""Analyze real Stage9 8-vs-32 client worker queue-isolation experiment.

The analysis **does not require or imply** that 32 threads reduce total
scheduled-arrival latency. It separates client scheduler delay from
post-dispatch SSE TTFT and samples vLLM server waiting/running gauges.
All paired comparisons use the identical frozen inputs and arrival schedule.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from llm_opt_lab.provenance import write_json_atomic
from llm_opt_lab.serving.bench_client import quantile

SOURCE_SHA = "9f497193c38c8d2bb00b6b78fd25d6b07bc66528"
FROZEN_SHA = "c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c"
WORKERS = (8, 32)
SEEDS = (42, 137)
CONDITIONS = (("label", 8), ("detailed", 2), ("detailed", 8))


def analyze(root: Path) -> dict:
    rows=[]
    pairs=[]
    for mode,rate in CONDITIONS:
        for seed in SEEDS:
            subset=[]
            for workers in WORKERS:
                name=f"{mode}_w{workers}_rate{rate}_seed{seed}.json"
                r=json.loads((root/name).read_text(encoding="utf-8"))
                cfg=r["config"]
                summary=r["summary"]
                if r.get("status")!="measured" or r.get("kind")!="real_vllm_stage8_multilingual_openloop":
                    raise ValueError(f"{name}: missing validated real vLLM SSE data")
                if r["environment"]["git_commit"]!=SOURCE_SHA or cfg["frozen_request_sha256"]!=FROZEN_SHA:
                    raise ValueError(f"{name}: source or input provenance mismatch")
                if (cfg["max_client_workers"],cfg["seed"],cfg["mode"],cfg["rate_per_second"])!=(workers,seed,mode,rate):
                    raise ValueError(f"{name}: condition mismatch")
                if cfg.get("schedule_kind")!="fixed_window" or cfg.get("output_cap_tokens")!=128:
                    raise ValueError("nonmatched experiment policy/cap")
                if abs(r["offered_arrival_rate_empirical_per_second"]-rate)>1e-8:
                    raise ValueError("arrival schedules not matched in empirical offered rate")
                if summary["succeeded"]!=24 or summary["failed"]!=0:
                    raise ValueError("missing or failed actual requests")
                if summary["usage_coverage"]!=1. or summary["prompt_usage_coverage"]!=1.:
                    raise ValueError("actual provider tokens unavailable")
                if not r["all_actual_input_tokens_in_bounds"] or r["length_gates"]["status"]!="pass":
                    raise ValueError("input/output length gate invalid")
                if len(r["requests"])!=24:
                    raise ValueError("missing per-request arrival/source evidence")
                ids=[x["request_id"] for x in r["requests"]]
                if len(set(ids))!=24 or len(set(x["language"] for x in r["requests"]))!=3:
                    raise ValueError("corpus identities/languages invalid")
                if any(x["status"]!="ok" or x.get("prompt_tokens") is None
                       or x.get("completion_tokens") is None
                       or x.get("ttft_from_due_ms") is None or x["client_executor_wait_ms"]<0
                       for x in r["requests"]):
                    raise ValueError("per-request timing/token trace incomplete")
                trace=[(x["request_id"],x["prompt_sha256"],x["scheduled_offset_s"]) for x in r["requests"]]
                arrival=[x["ttft_from_due_ms"] for x in r["requests"]]
                if abs(quantile(arrival,.95)-r["p95_ttft_from_scheduled_arrival_ms"])>1e-5:
                    raise ValueError("arrival TTFT summary does not match raw records")
                q=r["server_request_queue_gauges"]
                if not q["available"] or q["samples"]<1:
                    raise ValueError("vLLM server queue gauges unavailable")
                for k in ("max_server_waiting_requests","max_server_running_requests","mean_server_waiting_requests"):
                    if q[k] is None or q[k]<0 or not math.isfinite(q[k]):
                        raise ValueError("invalid sampled real server waiting/running gauges")
                datum={
                    "mode":mode,"rate":rate,"seed":seed,"client_workers":workers,
                    "source_file":name,"schedule":cfg["request_arrival_schedule_s"],
                    "trace":trace,
                    "p95_arrival_ttft_ms":r["p95_ttft_from_scheduled_arrival_ms"],
                    "p95_dispatch_ttft_ms":summary["p95_ttft_ms"],
                    "p95_client_executor_wait_ms":r["p95_executor_queue_wait_ms"],
                    "max_sampled_server_waiting_requests":q["max_server_waiting_requests"],
                    "max_sampled_server_running_requests":q["max_server_running_requests"],
                    "server_samples":q["samples"],
                    "output_tokens_per_second":summary["output_tokens_per_second"],
                    "completed_requests_per_second":r["completed_requests_per_second"],
                    "accepted":summary["succeeded"],"rejected":0,"errors":summary["failed"],
                    "actual_mean_completion_tokens":r["length_gates"]["realized_mean"],
                }
                rows.append(datum)
                subset.append(datum)
            before,after=subset
            if before["client_workers"]!=8 or after["client_workers"]!=32:
                raise ValueError("worker pair ordered incorrectly")
            if before["trace"]!=after["trace"] or before["schedule"]!=after["schedule"]:
                raise ValueError("A/B workload/seed/scheduled times differ")
            pairs.append({
                "mode":mode,"target_offered_rate":rate,"seed":seed,
                "workers_8":{
                    k:v for k,v in before.items() if k not in ("trace","schedule")
                },
                "workers_32":{
                    k:v for k,v in after.items() if k not in ("trace","schedule")
                },
                "p95_client_wait_reduction_ms":(
                    before["p95_client_executor_wait_ms"]-after["p95_client_executor_wait_ms"]
                ),
                "p95_arrival_ttft_reduction_ms":(
                    before["p95_arrival_ttft_ms"]-after["p95_arrival_ttft_ms"]
                ),
                "p95_dispatch_ttft_change_ms":(
                    after["p95_dispatch_ttft_ms"]-before["p95_dispatch_ttft_ms"]
                ),
                "server_max_waiting_change":(
                    after["max_sampled_server_waiting_requests"]-
                    before["max_sampled_server_waiting_requests"]
                ),
                "throughput_on_32_over_8":(
                    after["output_tokens_per_second"]/before["output_tokens_per_second"]
                    if before["output_tokens_per_second"]>0 else None
                ),
            })
    return {
        "status":"pass",
        "kind":"stage9_real_gpu_worker_vs_server_queue_isolation",
        "source_commit":SOURCE_SHA,
        "frozen_request_sha256":FROZEN_SHA,
        "real_requests":len(rows)*24,
        "case_count":len(rows),
        "matched_pairs":pairs,
        "not_a_new_gpu_measurement":True,
        "limitations":[
            "Server queue gauges sampled at 150ms; peaks may be missed",
            "Client thread worker saturation is not the same as server internal queue waiting",
            "Only 24 real requests and 2 seed schedules for each condition",
            "No new admission policy was run; only the load-generator worker cap changed",
            "Queue comparisons across sequential trials can be affected by GPU thermals and server state",
            "Real generation modes have different output length and prompt instruction overhead",
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
        print("STAGE9_ANALYSIS_PASS real_requests",result["real_requests"],flush=True)
        for pair in result["matched_pairs"]:
            print(pair["mode"],pair["target_offered_rate"],pair["seed"],
                  "client_wait_saved_ms",round(pair["p95_client_wait_reduction_ms"],1),
                  "arrival_TTFT_saved_ms",round(pair["p95_arrival_ttft_reduction_ms"],1),
                  "dispatch_TTFT_delta_ms",round(pair["p95_dispatch_ttft_change_ms"],1),
                  "sampled_server_waiting_delta",pair["server_max_waiting_change"])
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f"INVALID_STAGE9: {type(exc).__name__}: {exc}",flush=True)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
