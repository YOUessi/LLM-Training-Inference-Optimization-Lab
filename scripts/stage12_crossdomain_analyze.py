"""Validate previously unseen admission-policy seeds without overwriting originals.

Run after both approved GPU seed scripts complete. Report all offered
requests, explicit reject costs, language group outcomes and useful output
throughput. CPU post-processing only; never claims new GPU measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEEDS=(4093,5087)
MODES=("label","detailed")
ARMS=(("unlimited",250),("bounded",150),("bounded",400))
SOURCE_KEY="stage12-crossdomain-admission-comparison-v1"


def analyze(root: Path) -> dict:
    results=[]
    commits=set()
    for seed in SEEDS:
        for mode in MODES:
            baseline=None
            for policy,deadline in ARMS:
                p=root/f"stage12_seed_{seed}"/f"{mode}_{policy}_d{deadline}_seed{seed}.json"
                d=json.loads(p.read_text(encoding="utf-8"))
                if d.get("status")!="measured" or d.get("kind")!="stage10_real_client_admission_policy_study":
                    raise ValueError(f"invalid real model evidence: {p}")
                c=d["config"];s=d["summary"];r=d["requests"]
                expected_sha="fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04"
                if c.get("frozen_requests_sha256")!=expected_sha:
                    raise ValueError("independent-domain frozen input SHA mismatch")
                if (c["seed"],c["mode"],c["policy"],c["admission_deadline_ms"]) != (seed,mode,policy,deadline):
                    raise ValueError("case changed from approved held-out plan")
                if c["fixed_offered_rate"]!=8 or c["max_client_workers"]!=32:
                    raise ValueError("changed serving workload")
                if s["offered"]!=24 or s["accepted"]+s["rejected"]!=24:
                    raise ValueError("offered/accepted denominator broken")
                if s["completed"]+s["rejected"]!=24 or s["errors"]!=0:
                    raise ValueError("incomplete or silently failed work")
                if len(r)!=24 or len({x["request_id"] for x in r})!=24:
                    raise ValueError("request traces incomplete")
                if not d["server_queue_monitor"]["available"] or not d["gpu_kv_monitor"]["available"]:
                    raise ValueError("real server instrumentation missing")
                if d["accepted_output_length_gate"]["status"]!="pass":
                    raise ValueError("actual model completion gate failed")
                if s["by_language"]["en"]["offered"]!=24:
                    raise ValueError("English natural prose cohort incomplete")
                if any(s["by_language"][lang]["offered"]!=0 for lang in ("zh-cn","ja")):
                    raise ValueError("Stage12 claims wrong source language")
                commits.add(d["environment"]["git_commit"])
                trace=[(x["request_id"],x["prompt_sha256"],x["scheduled_offset_s"]) for x in r]
                if baseline is None:
                    baseline=trace
                elif trace!=baseline:
                    raise ValueError("policies did not see exact same arrivals/documents")
                results.append({
                    "seed":seed,"mode":mode,"policy":policy,"deadline_ms":deadline,
                    "offered":24,"completed":s["completed"],"rejected":s["rejected"],
                    "all_offered_250ms_slo":s["slo_all_offered"]["250"]["compliance_fraction"],
                    "all_offered_500ms_slo":s["slo_all_offered"]["500"]["compliance_fraction"],
                    "useful_output_tokens_per_second":s["useful_output_tokens_per_second"],
                    "rejection_reasons":s["rejection_reasons"],
                    "by_language":s["by_language"],
                    "server_peak_waiting_gauge":d["server_queue_monitor"]["max_waiting"],
                    "sample_count":d["server_queue_monitor"]["samples"],
                    "sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
                    "raw_file":str(p.relative_to(root)),
                })
    if len(commits)!=1:
        raise ValueError("held-out experiments originated from multiple Git commits")
    groups=[]
    for mode in MODES:
        for policy,deadline in ARMS:
            sample=[x for x in results if x["mode"]==mode and x["policy"]==policy and x["deadline_ms"]==deadline]
            if {x["seed"] for x in sample}!={4093,5087}:
                raise ValueError("missing independent held-out seeds")
            groups.append({
                "mode":mode,"policy":policy,"deadline_ms":deadline,
                "offered":48,
                "completed":sum(x["completed"] for x in sample),
                "rejected":sum(x["rejected"] for x in sample),
                "compliant_250ms":round(sum(24*x["all_offered_250ms_slo"] for x in sample)),
                "compliance_250ms_fraction":sum(x["all_offered_250ms_slo"] for x in sample)/2,
                "min_useful_output_tps":min(x["useful_output_tokens_per_second"] for x in sample),
                "max_useful_output_tps":max(x["useful_output_tokens_per_second"] for x in sample),
                "language_rejections":{
                    lang:sum(x["by_language"][lang]["rejected"] for x in sample)
                    for lang in ("en","zh-cn","ja")
                },
            })
    return {
        "schema":SOURCE_KEY,
        "status":"pass",
        "not_a_new_gpu_measurement":True,
        "source_git_commit":next(iter(commits)),
        "heldout_seeds":list(SEEDS),
        "offered":len(results)*24,
        "recorded_12_cases":len(results),
        "raw_cases":results,
        "two_seed_aggregate":groups,
        "source_domain":"English natural prose: five classic literary/natural-science genres, not Kubernetes",
        "frozen_domain_request_sha256":"fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04",
        "limitations":[
            "Independent source domain, but all source text is English classic prose; no production inference validation",
            "Two seeds and 24 offered requests per run do not establish tail-latency confidence intervals",
            "Client-side semaphore admission, not the vLLM engine scheduling algorithm",
            "A rejected request is never counted as a timely or completed success",
        ],
    }


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root",type=Path,default=Path("results"))
    p.add_argument("--out",type=Path,default=Path("results/stage12_crossdomain_analysis.json"))
    args=p.parse_args(argv)
    try:
        report=analyze(args.results_root)
        if args.out.exists():
            old=json.loads(args.out.read_text(encoding="utf-8"))
            if old!=report:
                raise ValueError("refusing to replace distinct held-out result")
        else:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print("STAGE12_ANALYSIS_PASS",report["offered"],"offered cases",report["recorded_12_cases"])
        for x in report["two_seed_aggregate"]:
            print("CASE",x["mode"],x["policy"],x["deadline_ms"],
                  "completed",x["completed"],"rejected",x["rejected"],
                  "SLO250",round(x["compliance_250ms_fraction"],3),
                  "min_useful_TPS",round(x["min_useful_output_tps"],1))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(f"STAGE12_ANALYSIS_BLOCKED {type(exc).__name__}: {exc}")
        return 2


if __name__=="__main__":
    raise SystemExit(main())
