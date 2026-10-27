"""vLLM mixed prefill/decode benchmark on frozen genuine literary prose.

Stage7 goal: 30 unique nonrepeated excerpts from 5 traced public-domain
English works, calibrated against real Qwen tokenizer. 4 input buckets x
2 generation modes x 4 concurrency settings x 2 independent seeded orders.

Do NOT infer output lengths from max_new_tokens: enforce real provider SSE
completion-token usage and a minimum mean length gate for long generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_opt_lab.provenance import environment, write_json_atomic
from llm_opt_lab.serving.bench_client import _endpoint, _request, summarize, quantile


# Character cutoffs are calibrated on the frozen 30 passages using the Qwen2.5
# tokenizer; prompt *token* validation uses real vLLM usage in every case.
BUCKETS: dict[str, tuple[int,int,int]] = {
    "short": (550,110,240),
    "medium": (1700,345,580),
    "long": (3900,740,1140),
    "very_long": (6000,1120,1720),
}

GENERATION: dict[str, dict[str, Any]] = {
    "short": {
        "max_new_tokens": 32,
        "min_mean_generated_tokens": 8,
        "response_instruction": (
            "Summarize the excerpt in exactly one informative short sentence."
        ),
    },
    "long": {
        "max_new_tokens": 128,
        "min_mean_generated_tokens": 85,
        "response_instruction": (
            "Produce a detailed numbered analysis of at least 20 separate "
            "observations grounded in the passage. Explain what the text "
            "describes, who or what it discusses, setting, evidence, uncertainty "
            "and implications. Continue with multiple observations instead of "
            "ending after one short sentence; use the available output budget."
        ),
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8026")
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", type=Path, default=Path("data/stage7_public_domain_passages.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/stage7_public_domain_manifest.json"))
    p.add_argument("--mode", choices=tuple(GENERATION), required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.concurrency not in (1,2,4,8):
        p.error("concurrency must be one of 1/2/4/8")
    if a.seed not in (42,137):
        p.error("seed must be one of frozen 42/137")
    if not 0 <= a.warmup <= 2:
        p.error("warmup must be between 0 and 2")
    return a


def load_corpus(path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    b = path.read_bytes()
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(b).hexdigest()!=manifest["corpus_jsonl_sha256"]:
        raise ValueError("corpus content SHA256 not the frozen manifest")
    rows=[json.loads(line) for line in b.decode("utf-8").splitlines()]
    if len(rows)!=30 or len(set(row["doc_id"] for row in rows))!=30:
        raise ValueError("corpus must include 30 unique documents")
    if len(set(row["genre"] for row in rows))!=5:
        raise ValueError("corpus must contain five genres")
    for row in rows:
        if hashlib.sha256(row["text"].encode("utf-8")).hexdigest()!=row["text_sha256"]:
            raise ValueError(f"passage hash mismatch: {row['doc_id']}")
    return rows,manifest


def truncate_sentence(text: str, chars: int) -> str:
    """Cut at last full-sentence boundary within the final 250 chars."""
    if len(text)<chars:
        raise ValueError("frozen excerpt shorter than requested length")
    t=text[:chars]
    candidates=[t.rfind(token) for token in (". ","! ","? ")]
    end=max(candidates)
    if end<chars-250:
        # Legal char-level prefix of a natural document, still nonrepeating.
        return t
    return t[:end+1]


def workload(rows: list[dict], *, seed: int) -> tuple[list[dict], list[dict]]:
    """Exactly 24 unique documents, 6 per length, 5 genres represented."""
    genres=sorted(set(row["genre"] for row in rows))
    by_genre: dict[str,list[dict]]=defaultdict(list)
    for row in rows:
        by_genre[row["genre"]].append(row)
    rng=random.Random(seed)
    for r in by_genre.values():
        rng.shuffle(r)
    items=[]
    j=0
    for bucket,(target,_,_) in BUCKETS.items():
        for k in range(6):
            genre=genres[j%len(genres)]
            j+=1
            if not by_genre[genre]:
                raise ValueError("not enough passages per genre for balancing")
            row=by_genre[genre].pop()
            passage=truncate_sentence(row["text"],target)
            prompt=(
                f"This excerpt is a passage of {row['genre']} from a public-domain book. "
                "Answer only from the supplied passage, not from external sources.\n"
                f"Excerpt:\n{passage}\n"
                f"Excerpt identifier: {row['doc_id']}"
            )
            items.append({
                "bucket":bucket,"genre":row["genre"],
                "doc_id":row["doc_id"],"ebook_id":row["source_ebook_id"],
                "passage_chars":len(passage),
                "prompt":prompt,
                "prompt_sha256":hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            })
    if len(items)!=24 or len({x["doc_id"] for x in items})!=24:
        raise ValueError("duplicate docs in a single real workload")
    rng.shuffle(items)
    remaining=[x for r in by_genre.values() for x in r]
    return items,remaining


def by_bucket(records: list[dict]) -> dict[str,dict]:
    result={}
    for key,(_,lower,upper) in BUCKETS.items():
        rs=[x for x in records if x["bucket"]==key]
        tokens=[x.get("prompt_tokens") for x in rs]
        good=(len(rs)==6 and all(x["status"]=="ok" for x in rs)
              and all(isinstance(n,int) and not isinstance(n,bool)
                      and lower<=n<=upper for n in tokens)
              and all(x.get("completion_tokens") is not None for x in rs))
        first=[x["ttft_ms"] for x in rs if x.get("ttft_ms") is not None]
        result[key]={
            "status":"pass" if good and len(first)==6 else "invalid",
            "sent":len(rs),"real_prompt_token_bounds":[lower,upper],
            "mean_actual_prompt_tokens":sum(tokens)/len(tokens) if tokens and all(isinstance(n,int) for n in tokens) else None,
            "min_actual_prompt_tokens":min(tokens) if tokens and all(isinstance(n,int) for n in tokens) else None,
            "max_actual_prompt_tokens":max(tokens) if tokens and all(isinstance(n,int) for n in tokens) else None,
            "p50_ttft_ms":quantile(first,.5),"p95_ttft_ms":quantile(first,.95),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    a=parse_args(argv)
    try:
        endpoint=_endpoint(a.base_url)
        corpus,manifest=load_corpus(a.corpus,a.manifest)
        items,remaining=workload(corpus,seed=a.seed)
        generation=GENERATION[a.mode]
        def send(i:int, x:dict):
            response=_request(
                i,endpoint=endpoint,model=a.model,prompt=x["prompt"],
                max_tokens=generation["max_new_tokens"],
                timeout=120.,seed=a.seed,
                response_instruction=generation["response_instruction"],
            )
            return {
                **response,
                "bucket":x["bucket"],"genre":x["genre"],
                "doc_id":x["doc_id"],"source_ebook_id":x["ebook_id"],
                "passage_chars":x["passage_chars"],
                "prompt_sha256":x["prompt_sha256"],
            }
        for i in range(a.warmup):
            source=remaining[i]
            x={
                "prompt":truncate_sentence(source["text"],550),
                "bucket":"warmup","genre":source["genre"],
                "doc_id":source["doc_id"],
                "ebook_id":source["source_ebook_id"],
                "passage_chars":550,
                "prompt_sha256":"not_used",
            }
            if send(-i-1,x)["status"]!="ok":
                raise RuntimeError("warmup failed; not allowed to claim measured throughput")
        start=time.perf_counter()
        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            futures=[pool.submit(send,i,x) for i,x in enumerate(items)]
            records=sorted((f.result() for f in as_completed(futures)),
                           key=lambda x:x["index"])
        wall=time.perf_counter()-start
        summary=summarize(records,wall)
        grouped=by_bucket(records)
        generated=[r["completion_tokens"] for r in records
                   if isinstance(r.get("completion_tokens"),int)]
        mean_generated=sum(generated)/len(generated) if generated else None
        successful=(
            summary["succeeded"]==24 and summary["failed"]==0
            and summary["prompt_usage_coverage"]==1.0
            and summary["usage_coverage"]==1.0
            and all(v["status"]=="pass" for v in grouped.values())
            and mean_generated is not None
            and mean_generated>=generation["min_mean_generated_tokens"]
        )
        outcome={
            "status":"measured" if successful else "invalid_or_blocked",
            "kind":"stage7_natural_public_domain_vllm_mixed_context",
            "engine":"vllm",
            "expected_server_prefix_cache":"off",
            "not_production_traffic":True,
            "environment":environment(),
            "config":{
                "model":a.model,"seed":a.seed,"concurrency":a.concurrency,
                "mode":a.mode,
                "max_new_tokens":generation["max_new_tokens"],
                "min_mean_generated_tokens":generation["min_mean_generated_tokens"],
                "response_instruction_sha256":hashlib.sha256(
                    generation["response_instruction"].encode("utf-8")
                ).hexdigest(),
                "corpus_sha256":manifest["corpus_jsonl_sha256"],
                "book_count":5,"source_document_count":30,
                "selected_unique_passages":24,
                "warmup_requests":a.warmup,
                "genre_names":list(sorted(set(x["genre"] for x in corpus))),
            },
            "summary":summary,
            "generated_tokens_min":min(generated) if generated else None,
            "generated_tokens_max":max(generated) if generated else None,
            "generated_tokens_mean":mean_generated,
            "bucket_statistics":grouped,
            "requests":records,
        }
        write_json_atomic(a.output,outcome)
        print("STAGE7_CASE",a.mode,"concurrency",a.concurrency,"seed",a.seed,
              "status",outcome["status"],
              "success",summary["succeeded"],
              "p95 TTFT",round(summary["p95_ttft_ms"] or 0,2),
              "out_tok/s",round(summary["output_tokens_per_second"] or 0,1),
              "actual_output_tokens_mean",round(mean_generated or 0,1),flush=True)
        for key,group in grouped.items():
            print(" ",key,group["status"],
                  "input_tokens",group["min_actual_prompt_tokens"],
                  group["max_actual_prompt_tokens"],flush=True)
        return 0 if successful else 1
    except (OSError,KeyError,TypeError,ValueError,RuntimeError) as exc:
        print(f"STAGE7 BLOCKED: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
