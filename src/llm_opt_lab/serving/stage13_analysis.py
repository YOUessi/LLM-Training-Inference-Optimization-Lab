"""Independent CPU analysis of Stage13 REAL 768-offer adaptive vLLM trial.

Do not claim an adaptive scheduler win without passing all-offered SLO
utility + throughput/reject predeclared constraints. Require source Git
SHA and actual frozen prompt/schedule identity for all policy comparisons.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter,defaultdict
from pathlib import Path

from llm_opt_lab.provenance import write_json_atomic
from llm_opt_lab.serving.stage10_admission import accounting,offered_slo_compliance
from llm_opt_lab.serving.stage13_mixed_bench import DOMAINS,RATES,POLICIES,make_offers
from llm_opt_lab.serving.stage8_arrivals import frozen_requests,schedule

FROZEN_CORPORA={
    "k8s":"c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c",
    "prose":"fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04",
}
UTILITY_REJECTION_PENALTY=.50
UTILITY_THROUGHPUT_WEIGHT=.20
CANDIDATE_MIN_GAIN=.02
CANDIDATE_MIN_THROUGHPUT_RATIO=.80
CANDIDATE_MAX_REJECTION_FRACTION=.20


def analyze(root:Path, expected_source_sha:str)->dict:
    if len(expected_source_sha)!=40:
        raise ValueError("expected exact full source Git SHA")
    raw=[]
    comparison=[]
    total=0
    for domain,(source,source_meta,seeds) in DOMAINS.items():
        originals,meta=frozen_requests(source,source_meta)
        if meta["request_jsonl_sha256"]!=FROZEN_CORPORA[domain]:
            raise ValueError("wrong corpus model input SHA")
        for seed in seeds:
            expected=make_offers(originals,seed)
            for rate in RATES:
                fixed_schedule=schedule(seed,rate,24,kind="fixed_window")
                series=[]
                for policy in POLICIES:
                    p=root/f"stage13_{domain}_seed{seed}"/f"rate{int(rate)}_{policy}_seed{seed}.json"
                    r=json.loads(p.read_text(encoding="utf-8"))
                    c=r["config"]
                    if r.get("status")!="measured" or r.get("kind")!="stage13_mixed_real_vllm_adaptive_admission":
                        raise ValueError(f"not accepted actual real model evidence: {p}")
                    if r["environment"]["git_commit"]!=expected_source_sha:
                        raise ValueError("source Git revision mismatch")
                    if (c["domain"],c["seed"],c["rate_per_second"],c["policy"])!=(domain,seed,rate,policy):
                        raise ValueError("experimental factorial cell mismatch")
                    if c["frozen_requests_sha256"]!=FROZEN_CORPORA[domain]:
                        raise ValueError("source text not frozen")
                    if c["max_client_workers"]!=32 or c["server_max_num_seqs"]!=8:
                        raise ValueError("wrong server/client concurrency")
                    if c["mode_count"]!={"label":12,"detailed":12}:
                        raise ValueError("invalid mixed workload")
                    if c["arrival_schedule_offsets_s"]!=fixed_schedule:
                        raise ValueError("policy measured with changed arrival schedule")
                    records=r["requests"]
                    if len(records)!=24 or len({x["request_id"] for x in records})!=24:
                        raise ValueError("missing or duplicated original offers")
                    expected_tuples=[(x["request_id"],x["prompt_sha256"],x["output_mode"],
                                      x["target_prompt_tokens"]) for x in expected]
                    actual_tuples=[(x["request_id"],x["prompt_sha256"],x["output_mode"],
                                    x["target_prompt_tokens"]) for x in records]
                    if actual_tuples!=expected_tuples:
                        raise ValueError("model prompt/task identity changed between policies")
                    if any(abs(x["scheduled_offset_s"]-fixed_schedule[i])>1e-8
                           for i,x in enumerate(records)):
                        raise ValueError("per-request offered time changed")
                    if any(x["status"] not in ("ok","rejected") for x in records):
                        raise ValueError("unknown/missing/error request status")
                    if any(x["status"]=="ok" and (
                            not isinstance(x.get("prompt_tokens"),int)
                            or not isinstance(x.get("completion_tokens"),int)
                            or abs(x["prompt_tokens"]-x["target_prompt_tokens"])>95)
                           for x in records):
                        raise ValueError("real model provider token usage invalid")
                    for x in records:
                        decision=x.get("policy_decision")
                        if not isinstance(decision,dict) or decision.get("policy_name")!=policy:
                            raise ValueError("missing or mismatched scheduling decision trace")
                        if policy=="adaptive" and (
                            decision["predicted_service_ms"]<=0 or
                            not 100<=decision["dynamic_admission_deadline_ms"]<=400 or
                            not 0<=decision["active_at_arrival"]<=8 or
                            not 0<=decision["waiting_at_arrival"]<=4):
                            raise ValueError("adaptive demand/queue state trace invalid")
                        if x["status"]=="rejected" and x.get("rejection_reason") not in (
                            "queue_full","deadline_expired"
                        ):
                            raise ValueError("rejection reason absent")
                    s=r["summary"]
                    if s!=accounting(records,r["elapsed_s"]):
                        raise ValueError("original all-offered metrics cannot be recomputed")
                    if s["offered"]!=24 or s["errors"]!=0 or s["completed"]+s["rejected"]!=24:
                        raise ValueError("missing/errored offered requests")
                    for threshold in (100.,250.,500.,1000.):
                        if s["slo_all_offered"][str(int(threshold))] != offered_slo_compliance(records,threshold):
                            raise ValueError("misreported all-offered SLO numerator/denominator")
                    if any(x["status"]=="ok" and x["ttft_from_due_ms"] is None for x in records):
                        raise ValueError("successful request missing first Token trace")
                    if not r["server_queue_monitor"]["available"] or r["server_queue_monitor"]["samples"]<1:
                        raise ValueError("actual vLLM server queue metrics not measured")
                    if not r["kv_cache_monitor"]["available"]:
                        raise ValueError("real KV cache use missing")
                    if r["gate_final_state"]["active"]!=0 or r["gate_final_state"]["waiters"]!=0:
                        raise ValueError("gate leaked inflight capacity after test")
                    modes=r["actual_mode_summaries"]
                    for mode in ("label","detailed"):
                        offered=[x for x in records if x["output_mode"]==mode]
                        if len(offered)!=12:
                            raise ValueError("mixed task ratio changed")
                        if modes[mode]["offered"]!=12:
                            raise ValueError("mode-level accounting changed")
                        if modes[mode]["length_gate"]["status"] not in ("pass","all_rejected"):
                            raise ValueError("completed tokens fail length gate")
                    for lang in set(x["language"] for x in records):
                        if s["by_language"][lang]["offered"]!=sum(x["language"]==lang for x in records):
                            raise ValueError("language offer accounting inconsistent")

                    total+=24
                    entry={
                        "domain":domain,"seed":seed,"rate":rate,
                        "policy":policy,
                        "offered":24,"completed":s["completed"],"rejected":s["rejected"],
                        "slo250":s["slo_all_offered"]["250"]["compliance_fraction"],
                        "useful_tps":s["useful_output_tokens_per_second"],
                        "p95_client_admission_wait_ms":s["p95_admission_queue_wait_ms"],
                        "sampled_max_server_waiting":r["server_queue_monitor"]["max_waiting"],
                        "by_mode":modes,
                        "by_language":s["by_language"],
                        "topic_offers":dict(Counter(x["topic"] for x in records)),
                        "topic_rejects":dict(Counter(x["topic"] for x in records if x["status"]=="rejected")),
                        "raw_file":str(p.relative_to(root)),
                    }
                    series.append(entry)
                    raw.append(entry)
                baseline=series[0]
                if baseline["policy"]!="unlimited" or baseline["useful_tps"]<=0:
                    raise ValueError("unlimited reference missing usable throughput")
                utility={}
                for x in series:
                    throughput_ratio=x["useful_tps"]/baseline["useful_tps"]
                    reject_fraction=x["rejected"]/24.
                    score=(x["slo250"]
                           -UTILITY_REJECTION_PENALTY*reject_fraction
                           +UTILITY_THROUGHPUT_WEIGHT*throughput_ratio)
                    utility[x["policy"]]={
                        "utility":score,
                        "relative_useful_tps":throughput_ratio,
                        "all_offered_slo250":x["slo250"],
                        "explicit_reject_fraction":reject_fraction,
                        "completed":x["completed"],
                    }
                adaptive=utility["adaptive"]
                baseline_util=utility["unlimited"]["utility"]
                best_static=max((utility["fixed150"]["utility"],
                                 utility["fixed400"]["utility"]))
                candidate_win=(
                    adaptive["utility"]>=baseline_util+CANDIDATE_MIN_GAIN
                    and adaptive["relative_useful_tps"]>=CANDIDATE_MIN_THROUGHPUT_RATIO
                    and adaptive["explicit_reject_fraction"]<=CANDIDATE_MAX_REJECTION_FRACTION
                )
                comparison.append({
                    "domain":domain,"seed":seed,"offered_rate":rate,
                    "paired_policies":utility,
                    "adaptive_minus_unlimited":adaptive["utility"]-baseline_util,
                    "adaptive_minus_best_fixed":adaptive["utility"]-best_static,
                    "candidate_win_vs_unlimited":candidate_win,
                    "best_policy_by_predeclared_utility":max(
                        utility,key=lambda k:utility[k]["utility"]),
                })
    if total!=768 or len(raw)!=32 or len(comparison)!=8:
        raise ValueError("full Stage13 32-case factorial experiment not measured")
    aggregated=[]
    for domain in DOMAINS:
        for rate in RATES:
            pairs=[x for x in comparison if x["domain"]==domain and x["offered_rate"]==rate]
            if len(pairs)!=2:
                raise ValueError("two seed comparisons missing")
            aggregated.append({
                "domain":domain,"offered_rate":rate,
                "seeds":[x["seed"] for x in pairs],
                "adaptive_candidate_wins":sum(x["candidate_win_vs_unlimited"] for x in pairs),
                "adaptive_delta_utility_range":[min(x["adaptive_minus_unlimited"] for x in pairs),
                                                 max(x["adaptive_minus_unlimited"] for x in pairs)],
                "adaptive_delta_vs_best_static_range":[min(x["adaptive_minus_best_fixed"] for x in pairs),
                                                       max(x["adaptive_minus_best_fixed"] for x in pairs)],
                "best_utility_policy_by_seed":[x["best_policy_by_predeclared_utility"] for x in pairs],
                "per_policy_all_offered_slo250":{
                    policy: [x["paired_policies"][policy]["all_offered_slo250"] for x in pairs]
                    for policy in POLICIES
                },
            })
    return {
        "status":"pass",
        "kind":"stage13_real_mixed_llm_adaptive_admission_analysis",
        "not_a_new_gpu_run":True,
        "source_commit":expected_source_sha,
        "replayed_offered_requests":total,
        "real_case_count":len(raw),
        "paired_scenarios":comparison,
        "per_case":raw,
        "two_seed_summary":aggregated,
        "predeclared_utility":{
            "formula":"all-offered-SLO250 - 0.50*reject_fraction + 0.20*(useful_TPS / paired_unlimited_useful_TPS)",
            "candidate_min_gain":CANDIDATE_MIN_GAIN,
            "minimum_relative_tps":CANDIDATE_MIN_THROUGHPUT_RATIO,
            "maximum_reject_fraction":CANDIDATE_MAX_REJECTION_FRACTION,
        },
        "scientific_limitations":[
            "Only 2 seeds/domain, previously seen source domains, 24 offers/case",
            "Client proxy mechanism is not a vLLM internal scheduler/engine optimization",
            "Adaptive combines online service-demand, deadline and priority; no separate ablations",
            "No confidence intervals or production-service guarantees from two small seeds",
            "All-offered 250ms timely first Token, useful output throughput and rejected load all reported",
            "This is one 1.5B model / RTX4090 laptop GPU; do not generalize to other model/hardware",
        ],
    }


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir",type=Path,default=Path("results"))
    p.add_argument("--expected-source-commit",required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        analysis=analyze(a.raw_dir,a.expected_source_commit)
        write_json_atomic(a.out,analysis)
        print("STAGE13_ANALYSIS_ACCEPTED",analysis["replayed_offered_requests"],
              "real offers /",analysis["real_case_count"],"cases")
        for x in analysis["two_seed_summary"]:
            print(x["domain"],x["offered_rate"],
                  "candidate_wins",x["adaptive_candidate_wins"],"/2",
                  "delta_util",tuple(round(y,3) for y in x["adaptive_delta_utility_range"]),
                  "best",x["best_utility_policy_by_seed"])
        return 0
    except (OSError,KeyError,ValueError,TypeError) as exc:
        print(f"STAGE13_ANALYSIS_INVALID: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
