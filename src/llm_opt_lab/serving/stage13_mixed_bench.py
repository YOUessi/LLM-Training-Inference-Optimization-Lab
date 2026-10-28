"""Stage13 real mixed-length workload, comparing NEW adaptive admission policy.

Same 24 frozen input identities and arrival offsets for direct baseline,
static 150/400ms admission, and age-fair demand-aware queue. Each request
is assigned a deterministic label/detailed task after seeded shuffle.
Output length, request state, and server gauge values come from REAL SSE.

Every offered request, including rejected and errored, stays in the
denominator. No acceptance-only P95 is promoted as an all-offered metric.
The new mechanism is a CLIENT PROXY, not a modified vLLM engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

from llm_opt_lab.provenance import environment,write_json_atomic
from llm_opt_lab.serving.bench_client import _endpoint,_request,quantile
from llm_opt_lab.serving.stage8_arrivals import (
    frozen_requests,output_length_checks,schedule,sample_kv_metrics,
)
from llm_opt_lab.serving.stage10_admission import accounting
from llm_opt_lab.serving.stage13_policy import AdmissionGate

DOMAINS={
    "k8s":(
        Path("data/stage8_frozen_requests.jsonl"),
        Path("data/stage8_frozen_requests_manifest.json"),
        (6101,6109),
    ),
    "prose":(
        Path("data/stage12_literary_requests.jsonl"),
        Path("data/stage12_literary_requests_manifest.json"),
        (7201,7209),
    ),
}
RATES=(8.,16.)
POLICIES=("unlimited","fixed150","fixed400","adaptive")
CLIENT_WORKERS=32
SERVER_MAX_SEQS=8
OUTPUT_CAP=128
OFFERS_PER_CASE=24
OUTPUT_MODES=("label","detailed")
THRESHOLDS=(100.,250.,500.,1000.)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url",default="http://127.0.0.1:8026")
    p.add_argument("--model",required=True)
    p.add_argument("--domain",choices=tuple(DOMAINS),required=True)
    p.add_argument("--rate",type=float,choices=RATES,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--policy",choices=POLICIES,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    if a.seed not in DOMAINS[a.domain][2]:
        p.error("seed not in predeclared held-out set for this domain")
    return a


def make_offers(rows:list[dict],seed:int) -> list[dict]:
    items=list(rows)
    random.Random(seed).shuffle(items)
    if len(items)!=OFFERS_PER_CASE or len({x["request_id"] for x in items})!=24:
        raise ValueError("24 distinct frozen requests required")
    result=[]
    for i,row in enumerate(items):
        kind=OUTPUT_MODES[i%2]
        result.append({
            **row,
            "output_mode":kind,
            "target_prompt_tokens":row["target_prompt_tokens"],
        })
    assert sum(x["output_mode"]=="label" for x in result)==12
    assert sum(x["output_mode"]=="detailed" for x in result)==12
    return result


def group_modes(records:list[dict]) -> dict:
    result={}
    for mode in OUTPUT_MODES:
        group=[x for x in records if x["output_mode"]==mode]
        done=[x for x in group if x["status"]=="ok"]
        if len(group)!=12:
            raise ValueError("not 12 offered requests per output mode")
        gate=(output_length_checks(mode,done)
              if done else {"status":"all_rejected","realized_mean":None})
        result[mode]={
            "offered":len(group),
            "completed":len(done),
            "rejected":sum(x["status"]=="rejected" for x in group),
            "errors":sum(x["status"]=="error" for x in group),
            "length_gate":gate,
            "p95_timed_ttft_completed_ms":quantile(
                [x["ttft_from_due_ms"] for x in done],.95
            ),
            "completed_output_tokens":sum(x["completion_tokens"] for x in done),
        }
    return result


def main(argv=None) -> int:
    a=parse_args(argv)
    try:
        endpoint=_endpoint(a.base_url)
        frozen,manifest=frozen_requests(*DOMAINS[a.domain][:2])
        items=make_offers(frozen,a.seed)
        instructions=manifest["generation_instructions"]
        if not all(k in instructions for k in OUTPUT_MODES):
            raise ValueError("frozen source missing mode instructions")
        warm=_request(-1,endpoint=endpoint,model=a.model,
                      prompt=items[-1]["prompt_text"],max_tokens=OUTPUT_CAP,
                      seed=a.seed,timeout=100.,
                      response_instruction=instructions["detailed"])
        if warm["status"]!="ok":
            raise RuntimeError("real model warmup failed")
        gate=AdmissionGate(a.policy)
        offsets=schedule(a.seed,a.rate,OFFERS_PER_CASE,kind="fixed_window")
        stop=threading.Event()
        kv_samples=[]
        queue_samples=[]
        sampler=threading.Thread(
            target=sample_kv_metrics,
            args=(a.base_url,stop,kv_samples,queue_samples),daemon=True,
        )
        sampler.start()
        start=time.perf_counter()+.1

        def one(i:int,item:dict,due:float)->dict:
            now=time.perf_counter()
            executor_wait=max(0.,now-due)*1000.
            mode=item["output_mode"]
            predicted_input=item["calibrated_prompt_tokens"][mode]
            gate_started=time.perf_counter()
            decision=gate.acquire(mode,predicted_input,due)
            wait_ms=max(0.,time.perf_counter()-gate_started)*1000.
            common={
                "index":i,
                "request_id":item["request_id"],
                "language":item["language"],"topic":item["topic"],
                "bucket":item["bucket"],"output_mode":mode,
                "prompt_sha256":item["prompt_sha256"],
                "target_prompt_tokens":item["target_prompt_tokens"],
                "calibrated_prompt_tokens":predicted_input,
                "scheduled_offset_s":due-start,
                "client_executor_wait_ms":executor_wait,
                "admission_queue_wait_ms":wait_ms,
                "policy_decision":decision.as_dict(),
            }
            if not decision.accepted:
                return {
                    **common,"status":"rejected",
                    "rejection_reason":decision.reason,
                    "ttft_from_due_ms":None,"ttft_ms":None,
                    "prompt_tokens":None,"completion_tokens":None,
                    "latency_ms":None,
                }
            service_start=time.perf_counter()
            try:
                result=_request(i,endpoint=endpoint,model=a.model,
                                prompt=item["prompt_text"],
                                max_tokens=OUTPUT_CAP,
                                timeout=120.,seed=a.seed,
                                response_instruction=instructions[mode])
                result.update(common)
                result["ttft_from_due_ms"]=(
                    max(0.,service_start-due)*1000.+result["ttft_ms"]
                    if result.get("ttft_ms") is not None else None
                )
                return result
            finally:
                duration=(time.perf_counter()-service_start)*1000.
                if a.policy!="unlimited":
                    gate.release(mode,duration)

        try:
            with ThreadPoolExecutor(max_workers=CLIENT_WORKERS) as pool:
                tasks=[]
                for i,(offset,item) in enumerate(zip(offsets,items)):
                    due=start+offset
                    time.sleep(max(0.,due-time.perf_counter()))
                    tasks.append(pool.submit(one,i,item,due))
                records=sorted(
                    (f.result() for f in as_completed(tasks)),
                    key=lambda x:x["index"]
                )
        finally:
            stop.set()
            sampler.join(timeout=3.)
        elapsed=time.perf_counter()-start
        totals=accounting(records,elapsed)
        by_mode=group_modes(records)
        valid=(
            totals["offered"]==24 and totals["errors"]==0
            and totals["completed"]+totals["rejected"]==24
            and totals["accepted"]==totals["completed"]
            and all(
                x["status"]!="ok" or (
                    isinstance(x.get("prompt_tokens"),int)
                    and abs(x["prompt_tokens"]-x["target_prompt_tokens"])<=95
                    and isinstance(x.get("completion_tokens"),int)
                )
                for x in records
            )
            and all(
                g["length_gate"]["status"] in ("pass","all_rejected")
                for g in by_mode.values()
            )
            and all(
                x["status"] in ("ok","rejected") for x in records
            )
        )
        if not valid:
            status="invalid_or_blocked"
        else:
            status="measured"
        payload={
            "status":status,
            "kind":"stage13_mixed_real_vllm_adaptive_admission",
            "not_vllm_internal_scheduler_modification":True,
            "environment":environment(),
            "config":{
                "domain":a.domain,"policy":a.policy,"seed":a.seed,
                "rate_per_second":a.rate,
                "frozen_requests_sha256":manifest["request_jsonl_sha256"],
                "max_client_workers":CLIENT_WORKERS,
                "server_max_num_seqs":SERVER_MAX_SEQS,
                "max_output_tokens":OUTPUT_CAP,
                "mixed_mode_assignment":"seeded shuffled 24 requests, alternating label and detailed",
                "mode_count":{"label":12,"detailed":12},
                "arrival_schedule":"seeded_fixed_duration_exponential_spacings",
                "arrival_schedule_offsets_s":offsets,
                "rate_is_not_unconditioned_poisson":True,
                "prior_service_ms":{"label":55.49,"detailed":1077.63},
                "policy_version":"stage13_age_fair_v1",
            },
            "summary":totals,
            "actual_mode_summaries":by_mode,
            "gate_final_state":gate.final_state(),
            "kv_cache_monitor":{
                "available":bool(kv_samples),"samples":len(kv_samples),
                "max_used_fraction":max((x["kv_usage"] for x in kv_samples),
                                        default=None),
            },
            "server_queue_monitor":{
                "available":bool(queue_samples),"samples":len(queue_samples),
                "max_waiting":max((x["waiting"] for x in queue_samples),
                                   default=None),
                "max_running":max((x["running"] for x in queue_samples),
                                   default=None),
            },
            "offered_sampled_actual_rate":(len(offsets)-1)/offsets[-1],
            "requests":records,
            "limitations":[
                "One GPU and Qwen2.5-1.5B FP16; no vLLM engine changes",
                "Online heuristic trained from prior fixed observations, not a learned optimal policy",
                "Capacity/deadline and short-job-first aging bundled into one algorithm; separate ablation not yet run",
                "Two seeds, 24 offers/case; no production P95 confidence interval",
                "Every rejected offer is a first-token SLO failure",
                "Kubernetes and classic prose corpus are both previously seen domains; new arrival seeds are held out",
            ],
        }
        write_json_atomic(a.output,payload)
        print(
            "STAGE13",a.domain,a.rate,a.seed,a.policy,
            "status",status,"offered",totals["offered"],
            "completed",totals["completed"],"rejected",totals["rejected"],
            "SLO250",round(totals["slo_all_offered"]["250"]["compliance_fraction"],3),
            "TPS",round(totals["useful_output_tokens_per_second"],1),
            "client_queue_ms",round(totals["p95_admission_queue_wait_ms"] or 0,1),
            "server_max_waiting",payload["server_queue_monitor"]["max_waiting"],
            flush=True,
        )
        return 0 if valid else 1
    except (OSError,KeyError,ValueError,TypeError,RuntimeError) as exc:
        print(f"STAGE13_INVALID {type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
