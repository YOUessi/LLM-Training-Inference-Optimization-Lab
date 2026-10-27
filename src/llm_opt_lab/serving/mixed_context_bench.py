"""Real mixed-context-length vLLM prefill latency / throughput workload.

Heterogeneous prompt lengths are each measured using provider-reported
prompt_tokens. Per-length and overall latency are reported separately.
Prefix cache is expected disabled for prefill-isolated measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic
from llm_opt_lab.serving.bench_client import _endpoint, _request, quantile, summarize
from llm_opt_lab.serving.prefix_cache_bench import make_prompt


LENGTHS = {
    "short": (3, 85, 250),
    "medium": (14, 400, 650),
    "long": (35, 1050, 1400),
    "very_long": (47, 1450, 1850),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8026")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--repeats-per-length", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if not 1 <= a.concurrency <= 8:
        p.error("concurrency must lie in 1..8")
    if not 2 <= a.repeats_per_length <= 16:
        p.error("repeats-per-length must lie in 2..16")
    if not 1 <= a.max_new_tokens <= 128 or not 0 <= a.warmup <= 4:
        p.error("max tokens must be 1..128 and warmup 0..4")
    return a


def bucketize(records: list[dict], *, configurations: dict[str, tuple[int,int,int]]) -> dict:
    result = {}
    for key, (_,low,high) in configurations.items():
        rows=[r for r in records if r["length_bucket"]==key]
        good=[r for r in rows if r["status"]=="ok"]
        counts=[r.get("prompt_tokens") for r in good]
        valid=bool(good) and len(good)==len(rows) and all(
            isinstance(n,int) and not isinstance(n,bool) and low<=n<=high
            for n in counts
        )
        ttft=[r["ttft_ms"] for r in good if r["ttft_ms"] is not None]
        result[key]={
            "status":"pass" if valid and len(ttft)==len(good) else "invalid",
            "success":len(good),"sent":len(rows),
            "min_prompt_tokens":min(counts) if counts and all(isinstance(n,int) for n in counts) else None,
            "max_prompt_tokens":max(counts) if counts and all(isinstance(n,int) for n in counts) else None,
            "mean_prompt_tokens":sum(counts)/len(counts) if counts and all(isinstance(n,int) for n in counts) else None,
            "p50_ttft_ms":quantile(ttft,.5),
            "p95_ttft_ms":quantile(ttft,.95),
            "p95_latency_ms":quantile([r["latency_ms"] for r in good],.95),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        endpoint=_endpoint(a.base_url)
    except ValueError as exc:
        print(f"BLOCKED: {exc}",file=sys.stderr);return 2
    items=[]
    index=0
    for key,(repetitions,_,_) in LENGTHS.items():
        for _ in range(a.repeats_per_length):
            # Intentional unique identifier early in the prefix, and a seed
            # salted with concurrency. This avoids cross-case cache pollution.
            prompt=make_prompt(index,seed=a.seed+a.concurrency*1009,
                               repetitions=repetitions,mode="distinct")
            items.append((index,key,prompt))
            index+=1
    rnd=random.Random(a.seed)
    rnd.shuffle(items)

    def send(item):
        i,key,prompt=item
        response=_request(i,endpoint=endpoint,model=a.model,prompt=prompt,
                          max_tokens=a.max_new_tokens,timeout=120.,seed=a.seed,
                          response_instruction="Answer in one short technical sentence.")
        response["length_bucket"]=key
        return response

    if a.warmup:
        for x in items[:a.warmup]:
            if send(x)["status"]!="ok":
                print("BLOCKED: streaming model warmup failed",file=sys.stderr)
                return 2

    begin=time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        futures=[pool.submit(send,item) for item in items]
        records=sorted((f.result() for f in as_completed(futures)),key=lambda d:d["index"])
    wall=time.perf_counter()-begin
    grouped=bucketize(records,configurations=LENGTHS)
    summary=summarize(records,wall)
    good=summary["succeeded"]==len(items) and all(x["status"]=="pass" for x in grouped.values())
    result={
        "status":"measured" if good else "invalid_or_blocked",
        "kind":"vllm_mixed_context_real_generation_workload",
        "engine":"vllm",
        "prefix_caching_state_expected":"off",
        "environment":environment(),
        "config":{
            "model":a.model,"concurrency":a.concurrency,
            "seed":a.seed,"warmup":a.warmup,
            "repeats_per_length":a.repeats_per_length,
            "max_new_tokens":a.max_new_tokens,
            "prompt_template_sha256":hashlib.sha256(
                repr(LENGTHS).encode("utf-8")
            ).hexdigest(),
        },
        "summary":summary,
        "by_prompt_length":grouped,
        "requests":records,
    }
    write_json_atomic(a.output,result)
    print(f"MIXED_CONTEXT concurrency={a.concurrency} status={result['status']} "
          f"success={summary['succeeded']}/{summary['sent']} "
          f"p95_TTFT={summary['p95_ttft_ms']:.1f}ms "
          f"output_tps={summary['output_tokens_per_second']:.1f}")
    for name,group in grouped.items():
        print("  "+name,"input",group["mean_prompt_tokens"],
              "p95_TTFT",group["p95_ttft_ms"],"status",group["status"])
    return 0 if good else 1


if __name__=="__main__":
    raise SystemExit(main())
