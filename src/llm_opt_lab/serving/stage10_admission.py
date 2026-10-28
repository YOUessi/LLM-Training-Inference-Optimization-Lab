"""Real controlled client-side admission policy under fixed vLLM offered load.

Compare baseline direct-submit (32 client workers) with bounded (8 in-flight,
4 queue slots, 250ms admission deadline). Every OFFFERED request stays in
denominator, including rejected/expired ones. This is a client proxy policy,
not a modification of vLLM's built-in scheduler.

Only test against local, authorized model serving workloads.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_opt_lab.provenance import environment, write_json_atomic
from llm_opt_lab.serving.bench_client import _endpoint, _request, quantile
from llm_opt_lab.serving.stage8_arrivals import (
    frozen_requests, output_length_checks, sample_kv_metrics, schedule,
)

MODES=("label","detailed")
POLICIES=("unlimited","bounded")
SEEDS=(42,137)
OFFERED_RATE=8.
OFFERED_COUNT=24
SERVER_MAX_SEQS=8
MAX_CLIENT_WORKERS=32
BOUNDED_INFLIGHT=8
BOUNDED_WAITING=4
ADMISSION_DEADLINE_MS=250.
OUTPUT_CAP=128
SLO_MS=(100.,250.,500.,1000.)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True)
    p.add_argument("--base-url",default="http://127.0.0.1:8026")
    p.add_argument("--mode",choices=MODES,required=True)
    p.add_argument("--policy",choices=POLICIES,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--requests",type=Path,
                   default=Path("data/stage8_frozen_requests.jsonl"))
    p.add_argument("--manifest",type=Path,
                   default=Path("data/stage8_frozen_requests_manifest.json"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    if a.seed not in SEEDS:
        p.error("seed must be one of 42,137")
    return a


def offered_slo_compliance(records: list[dict], threshold_ms: float) -> dict:
    """Rejected, timeout, error, or late -> NOT compliant; no denominator gaming."""
    if threshold_ms<=0:
        raise ValueError("SLO threshold must be positive")
    offered=len(records)
    if not offered:
        raise ValueError("cannot score an empty offered workload")
    good=sum(
        x["status"]=="ok"
        and x.get("ttft_from_due_ms") is not None
        and x["ttft_from_due_ms"]<=threshold_ms
        for x in records
    )
    return {
        "threshold_ms":threshold_ms,
        "offered":offered,
        "compliant_completed":good,
        "not_compliant":offered-good,
        "compliance_fraction":good/offered,
    }


def accounting(records: list[dict], elapsed: float) -> dict:
    if len(records)!=OFFERED_COUNT:
        raise ValueError("offered request count must be 24")
    if len({x["request_id"] for x in records})!=OFFERED_COUNT:
        raise ValueError("duplicate or missing request IDs")
    accepted=[x for x in records if x["status"]!="rejected"]
    finished=[x for x in records if x["status"]=="ok"]
    rejected=[x for x in records if x["status"]=="rejected"]
    error=[x for x in records if x["status"]=="error"]
    if len(accepted)+len(rejected)!=OFFERED_COUNT:
        raise ValueError("offered/accepted/rejected accounting mismatch")
    output=[x.get("completion_tokens") for x in finished]
    if any(not isinstance(n,int) or n<0 for n in output):
        raise ValueError("accepted result missing real output Token usage")
    if any(x.get("prompt_tokens") is None for x in finished):
        raise ValueError("accepted result missing real prompt Tokens")
    if elapsed<=0:
        raise ValueError("elapsed must be positive")
    complete_ttft=[x["ttft_from_due_ms"] for x in finished
                    if x.get("ttft_from_due_ms") is not None]
    if len(complete_ttft)!=len(finished):
        raise ValueError("successful request missing real first-token timestamp")
    rejected_reasons=sorted({x.get("rejection_reason") for x in rejected})
    if any(r not in ("queue_full","deadline_expired") for r in rejected_reasons):
        raise ValueError("unknown rejection reason")
    return {
        "offered":OFFERED_COUNT,
        "accepted":len(accepted),
        "completed":len(finished),
        "rejected":len(rejected),
        "errors":len(error),
        "acceptance_fraction":len(accepted)/OFFERED_COUNT,
        "completed_fraction":len(finished)/OFFERED_COUNT,
        "rejection_reasons":{
            k:sum(x.get("rejection_reason")==k for x in rejected)
            for k in rejected_reasons
        },
        "useful_output_tokens":sum(output),
        "useful_output_tokens_per_second":sum(output)/elapsed,
        "p95_completed_only_arrival_ttft_ms":quantile(complete_ttft,.95),
        # With a reject, all-offered P95 cannot be observed as a finite
        # request completion time. Never silently compute accepted-only P95
        # and label it "all offered".
        "p95_all_offered_ttft_ms":(
            quantile(complete_ttft,.95)
            if len(finished)==OFFERED_COUNT else None
        ),
        "p95_executor_wait_ms":quantile(
            [x["client_executor_wait_ms"] for x in records],.95
        ),
        "p95_admission_queue_wait_ms":quantile(
            [x["admission_queue_wait_ms"] for x in records],.95
        ),
        "slo_all_offered":{
            str(int(limit)):offered_slo_compliance(records,limit)
            for limit in SLO_MS
        },
        "completion_rate_per_second":len(finished)/elapsed,
        "by_language":{
            lang:{
                "offered":sum(x["language"]==lang for x in records),
                "completed":sum(x["language"]==lang and x["status"]=="ok"
                                for x in records),
                "rejected":sum(x["language"]==lang and x["status"]=="rejected"
                               for x in records),
            }
            for lang in ("en","zh-cn","ja")
        },
    }


def main(argv: list[str] | None = None) -> int:
    a=parse_args(argv)
    try:
        endpoint=_endpoint(a.base_url)
        items,meta=frozen_requests(a.requests,a.manifest)
        random.Random(a.seed).shuffle(items)
        instruction=meta["generation_instructions"][a.mode]
        # Preflight model is excluded from offered count.
        warm=_request(-1,endpoint=endpoint,model=a.model,prompt=items[-1]["prompt_text"],
                      max_tokens=OUTPUT_CAP,timeout=90.,seed=a.seed,
                      response_instruction=instruction)
        if warm["status"]!="ok":
            raise RuntimeError("model warmup failed")
        offsets=schedule(a.seed,OFFERED_RATE,len(items),kind="fixed_window")
        inflight=threading.BoundedSemaphore(BOUNDED_INFLIGHT)
        waiting=threading.BoundedSemaphore(BOUNDED_WAITING)
        kv_samples:list[dict]=[]
        queue_samples:list[dict]=[]
        stop=threading.Event()
        sampler=threading.Thread(
            target=sample_kv_metrics,
            args=(a.base_url,stop,kv_samples,queue_samples),daemon=True,
        )
        sampler.start()
        start=time.perf_counter()+.1

        def one(index:int,item:dict,due:float):
            arrival_into_executor=time.perf_counter()
            executor_wait=max(0.,arrival_into_executor-due)*1000.
            admitted_at=arrival_into_executor
            reason=None
            acquired=False
            if a.policy=="bounded":
                if inflight.acquire(blocking=False):
                    acquired=True
                elif not waiting.acquire(blocking=False):
                    reason="queue_full"
                else:
                    try:
                        remaining=due+(ADMISSION_DEADLINE_MS/1000.)-time.perf_counter()
                        if remaining<=0 or not inflight.acquire(timeout=remaining):
                            reason="deadline_expired"
                        else:
                            acquired=True
                    finally:
                        waiting.release()
                admitted_at=time.perf_counter()
            admission_wait_ms=max(0.,admitted_at-arrival_into_executor)*1000.
            common={
                "index":index,
                "request_id":item["request_id"],
                "language":item["language"],"topic":item["topic"],
                "bucket":item["bucket"],
                "prompt_sha256":item["prompt_sha256"],
                "target_prompt_tokens":item["target_prompt_tokens"],
                "calibrated_prompt_tokens":item["calibrated_prompt_tokens"][a.mode],
                "scheduled_offset_s":due-start,
                "client_executor_wait_ms":executor_wait,
                "admission_queue_wait_ms":admission_wait_ms,
            }
            if reason is not None:
                return {
                    **common,"status":"rejected","rejection_reason":reason,
                    "ttft_from_due_ms":None,"ttft_ms":None,
                    "prompt_tokens":None,"completion_tokens":None,
                    "latency_ms":None,
                }
            try:
                sent=time.perf_counter()
                response=_request(index,endpoint=endpoint,model=a.model,
                                  prompt=item["prompt_text"],max_tokens=OUTPUT_CAP,
                                  timeout=120.,seed=a.seed,
                                  response_instruction=instruction)
                response.update(common)
                response["ttft_from_due_ms"]=(
                    max(0.,sent-due)*1000.+response["ttft_ms"]
                    if response.get("ttft_ms") is not None else None
                )
                return response
            finally:
                if acquired:
                    inflight.release()

        try:
            with ThreadPoolExecutor(max_workers=MAX_CLIENT_WORKERS) as pool:
                futures=[]
                for i,(offset,item) in enumerate(zip(offsets,items)):
                    due=start+offset
                    time.sleep(max(0.,due-time.perf_counter()))
                    futures.append(pool.submit(one,i,item,due))
                data=sorted((f.result() for f in as_completed(futures)),
                            key=lambda x:x["index"])
        finally:
            stop.set()
            sampler.join(timeout=3.)
        elapsed=time.perf_counter()-start
        results=accounting(data,elapsed)
        # A measured admission policy may reject requests, but must never
        # hide them, and completed ones must pass the actual input/output
        # token and generation length gates.
        successful=[x for x in data if x["status"]=="ok"]
        lengths=output_length_checks(a.mode,successful)
        valid=(
            results["offered"]==24
            and results["errors"]==0
            and results["accepted"]==results["completed"]
            and all(
                abs(x["prompt_tokens"]-x["target_prompt_tokens"])<=95
                for x in successful
            )
            and lengths["status"]=="pass"
            and all(x["status"] in ("ok","rejected") for x in data)
        )
        peak_wait=[x["waiting"] for x in queue_samples]
        peak_running=[x["running"] for x in queue_samples]
        total={
            "status":"measured" if valid else "invalid_or_blocked",
            "kind":"stage10_real_client_admission_policy_study",
            "not_vllm_internal_scheduler_modification":True,
            "environment":environment(),
            "config":{
                "policy":a.policy,"mode":a.mode,"seed":a.seed,
                "model":a.model,"fixed_offered_rate":OFFERED_RATE,
                "frozen_requests_sha256":meta["request_jsonl_sha256"],
                "max_client_workers":MAX_CLIENT_WORKERS,
                "server_max_num_seqs":SERVER_MAX_SEQS,
                "bounded_inflight":BOUNDED_INFLIGHT,
                "bounded_queue_slots":BOUNDED_WAITING,
                "admission_deadline_ms":ADMISSION_DEADLINE_MS,
                "output_cap_tokens":OUTPUT_CAP,
                "schedule":"seeded_fixed_duration_exponential_spacings",
                "schedule_offsets_s":offsets,
            },
            "elapsed_s":elapsed,
            "accepted_output_length_gate":lengths,
            "summary":results,
            "server_queue_monitor":{
                "available":bool(queue_samples),
                "samples":len(queue_samples),
                "max_waiting":max(peak_wait) if peak_wait else None,
                "max_running":max(peak_running) if peak_running else None,
            },
            "gpu_kv_monitor":{
                "available":bool(kv_samples),
                "samples":len(kv_samples),
                "max_usage_fraction":(
                    max(x["kv_usage"] for x in kv_samples)
                    if kv_samples else None
                ),
            },
            "requests":data,
            "limitations":[
                "Client-side admission policy; NOT vLLM engine modification",
                "Queue-slot semaphores are not a strict fair FIFO scheduler",
                "Rejected and error requests are SLO failures, not missing data",
                "Only 24 offers/seed and one model/GPU",
                "Real Prometheus gauge sampled at 150ms may miss spikes",
            ],
        }
        write_json_atomic(a.output,total)
        print(
            "STAGE10",a.policy,a.mode,a.seed,
            "status",total["status"],
            "offered",results["offered"],
            "completed",results["completed"],
            "rejected",results["rejected"],
            "compliance_250ms",
            results["slo_all_offered"]["250"]["compliance_fraction"],
            "useful_tokens_s",round(results["useful_output_tokens_per_second"],1),
            flush=True,
        )
        return 0 if valid else 1
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
        print(f"STAGE10 INVALID: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
