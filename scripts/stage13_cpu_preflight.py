"""CPU-only Stage13 research manifest and integrity gate for 10-min CI.

All source corpora, seeds, policy identities and experiment dimensions are
fixed before any real GPU results. GitHub CI cannot access Tang's GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_opt_lab.serving.stage8_arrivals import frozen_requests
from llm_opt_lab.serving.stage13_mixed_bench import DOMAINS,POLICIES,RATES,make_offers
from llm_opt_lab.serving.stage13_analysis import (
    FROZEN_CORPORA,UTILITY_REJECTION_PENALTY,
    UTILITY_THROUGHPUT_WEIGHT,CANDIDATE_MIN_GAIN,
)


def preflight(root:Path)->dict:
    corpora=[]
    for domain,(source,source_manifest,seeds) in DOMAINS.items():
        rows,meta=frozen_requests(root/source,root/source_manifest)
        if meta["request_jsonl_sha256"]!=FROZEN_CORPORA[domain]:
            raise ValueError("unapproved domain source hash")
        if len(rows)!=24 or len({x["request_id"] for x in rows})!=24:
            raise ValueError("frozen source has wrong request identity")
        for seed in seeds:
            assignment=make_offers(rows,seed)
            if len(assignment)!=24 or set(x["output_mode"] for x in assignment)!={"label","detailed"}:
                raise ValueError("mixed short-long request schedule invalid")
        corpora.append({"domain":domain,"sha256":meta["request_jsonl_sha256"],
                        "seeds":list(seeds),"frozen_requests":len(rows)})
    k8s={x["language"] for x in frozen_requests(
        root/DOMAINS["k8s"][0],root/DOMAINS["k8s"][1])[0]}
    prose={x["language"] for x in frozen_requests(
        root/DOMAINS["prose"][0],root/DOMAINS["prose"][1])[0]}
    if k8s!={"en","zh-cn","ja"} or prose!={"en"}:
        raise ValueError("source language cohort changed")
    if (len(POLICIES),len(RATES))!=(4,2):
        raise ValueError("Stage13 factorial condition count changed")
    if not (root/"docs/STAGE13_PROTOCOL.md").exists():
        raise FileNotFoundError("immutable preregistration missing")
    return {
        "schema":"stage13-real-adaptive-admission-preflight-v1",
        "status":"pass",
        "source_domains":corpora,
        "all_offered":2*2*2*4*24,
        "real_gpu_cases":32,
        "offered_rate_per_second":list(RATES),
        "policies":list(POLICIES),
        "mixed_task_modes":{"label":12,"detailed":12},
        "workload_aware_prior_label_ms":55.49,
        "workload_aware_prior_detailed_ms":1077.63,
        "utility_formula":"all-offered SLO250 - 0.50*rejection_fraction + 0.20*(useful_tps/paired_unlimited_tps)",
        "frozen_penalty_coefficients":{
            "rejected":UTILITY_REJECTION_PENALTY,
            "relative_throughput":UTILITY_THROUGHPUT_WEIGHT,
            "candidate_min_gain":CANDIDATE_MIN_GAIN,
        },
        "approved_gpu_jobs":[
            "stage13_k8s_seed6101","stage13_k8s_seed6109",
            "stage13_prose_seed7201","stage13_prose_seed7209",
        ],
        "approved_cpu_job":"stage13_all_analysis",
        "github_hosted_has_tang_gpu":False,
        "changes_vllm_internal_scheduler":False,
    }


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(argv)
    d=preflight(args.repo_root)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(d,indent=2)+"\n",encoding="utf-8")
    print("STAGE13_CPU_PREFLIGHT_PASS",d["all_offered"],"approved offered requests")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
