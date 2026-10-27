"""Classify prefix KV cache retained vs evicted at controlled block budgets.

Stage 5 expected eviction after 12 fillers. Stage 6 probes multiple capacities
and filler loads; retained cache is a valid measurement, not a failure.
Real Prometheus hit-token deltas are required; cannot infer from TTFT alone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic
from llm_opt_lab.serving.bench_client import _endpoint, _request
from llm_opt_lab.serving.kv_capacity_probe import fetch_counters, ratio_between
from llm_opt_lab.serving.prefix_cache_bench import make_prompt


STAGES=("cold","warm","after_pressure","after_rewarm")


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url",default="http://127.0.0.1:8026")
    p.add_argument("--model",required=True)
    p.add_argument("--blocks",type=int,required=True)
    p.add_argument("--fillers",type=int,required=True)
    p.add_argument("--seed",type=int,default=830)
    p.add_argument("--prefix-repetitions",type=int,default=35)
    p.add_argument("--max-new-tokens",type=int,default=16)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    if not 32 <= a.blocks <= 4096 or not 1 <= a.fillers <= 80:
        p.error("blocks 32..4096 and fillers 1..80 required")
    if not 8 <= a.prefix_repetitions <= 50 or not 1 <= a.max_new_tokens <= 64:
        p.error("invalid prompt/output dimensions")
    return a


def classify(ratios: dict[str,float]) -> dict:
    if set(ratios)!=set(STAGES):
        raise ValueError("missing required probe stages")
    if any(not 0 <= v <= 1 for v in ratios.values()):
        raise ValueError("invalid hit fraction")
    precondition=ratios["cold"]<=.10 and ratios["warm"]>=.75
    final_check=ratios["after_rewarm"]>=.75
    after=ratios["after_pressure"]
    category="evicted" if after<=.25 else ("retained" if after>=.75 else "partial")
    return {
        "stage_valid":precondition and final_check,
        "post_pressure_classification":category,
        "initial_miss_valid":ratios["cold"]<=.10,
        "immediate_hit_valid":ratios["warm"]>=.75,
        "replay_hit_valid":final_check,
        "post_pressure_hit_fraction":after,
    }


def main(argv=None):
    a=parse_args(argv)
    try:
        endpoint=_endpoint(a.base_url)
        anchor=make_prompt(0,seed=a.seed,repetitions=a.prefix_repetitions,mode="shared")
        events=[]
        def call(stage,prompt,index):
            before=fetch_counters(a.base_url)
            r=_request(index,endpoint=endpoint,model=a.model,prompt=prompt,
                       max_tokens=a.max_new_tokens,timeout=90,seed=a.seed,
                       response_instruction="Give a one-sentence technical summary.")
            after=fetch_counters(a.base_url)
            if r["status"]!="ok" or not isinstance(r.get("prompt_tokens"),int):
                raise RuntimeError(f"invalid actual model response at {stage}")
            if not 250 <= r["prompt_tokens"] <= 1700:
                raise ValueError(f"actual prompt length out of bounds: {r['prompt_tokens']}")
            d=ratio_between(before,after)
            events.append({
                "stage":stage,"index":index,"prompt_tokens":r["prompt_tokens"],
                "completion_tokens":r["completion_tokens"],
                "ttft_ms":r["ttft_ms"],"query_delta":d["queried_tokens"],
                "hit_delta":d["cache_hit_tokens"],
                "hit_fraction":d["cache_hit_fraction"],
            })
        call("cold",anchor,0)
        call("warm",anchor,0)
        for i in range(a.fillers):
            unique=make_prompt(100+i,seed=a.seed+123,
                               repetitions=a.prefix_repetitions,mode="distinct")
            call(f"filler_{i:03d}",unique,100+i)
        call("after_pressure",anchor,0)
        call("after_rewarm",anchor,0)
        classifications=classify({
            x["stage"]:x["hit_fraction"] for x in events if x["stage"] in STAGES
        })
        result={
            "status":"measured" if classifications["stage_valid"] else "invalid",
            "kind":"stage6_kv_retention_curve_single_case",
            "engine":"vllm",
            "not_a_model_quality_benchmark":True,
            "environment":environment(),
            "config":{"blocks":a.blocks,"fillers":a.fillers,
                      "seed":a.seed,"prefix_repetitions":a.prefix_repetitions,
                      "max_new_tokens":a.max_new_tokens,"model":a.model},
            "classification":classifications,
            "events":events,
            "limitation":"Eviction classified from prefix-hit loss; not a direct block table or replacement-policy inspection.",
        }
        write_json_atomic(a.output,result)
        print("KV_RETENTION",result["status"],a.blocks,a.fillers,
              classifications["post_pressure_classification"],
              "after_hit",round(classifications["post_pressure_hit_fraction"],3),flush=True)
        return 0 if result["status"]=="measured" else 1
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
