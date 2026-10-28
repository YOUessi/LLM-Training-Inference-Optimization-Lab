"""Prepare a pinned non-Kubernetes natural prose admission workload.

Reuse only already pinned US-public-domain Project Gutenberg Stage7 excerpts.
Freeze exact Qwen2.5 tokenizer-calibrated 24 requests from five books.
Run on Tang's isolated tokenizer environment; no training or GPU usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


MODEL_SNAPSHOT="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
LENGTHS=("short","medium","long","very_long")
TARGETS={"short":170,"medium":300,"long":470,"very_long":680}
INSTRUCTIONS={
    "label":"Name the main topic of the excerpt in a short noun phrase. No explanation.",
    "summary":"Summarize the passage in two informative sentences.",
    "detailed":"List at least 20 distinct numbered observations from the excerpt. Continue until the output limit.",
}


def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def assign(rows:list[dict])->list[dict]:
    if len(rows)!=30 or len({r["doc_id"] for r in rows})!=30:
        raise ValueError("require the immutable 30-excerpt Stage7 corpus")
    by_genre=defaultdict(list)
    for x in rows:
        by_genre[x["genre"]].append(x)
    if len(by_genre)!=5 or {len(v) for v in by_genre.values()}!={6}:
        raise ValueError("require 5 genres and six distinct passages per genre")
    genres=sorted(by_genre)
    for i,genre in enumerate(genres):
        random.Random(12000+i).shuffle(by_genre[genre])
    counters=Counter()
    jobs=[]
    for i in range(24):
        genre=genres[i%5]
        row=by_genre[genre][counters[genre]]
        counters[genre]+=1
        bucket=LENGTHS[i//6]
        jobs.append({
            **row,
            "language":"en",
            "topic":genre,
            "bucket":bucket,
            "target_prompt_tokens":TARGETS[bucket],
        })
    if len({x["doc_id"] for x in jobs})!=24 or set(counters.values())!={4,5}:
        raise AssertionError("selection is not nonoverlapping and genre-balanced")
    return jobs


def body_prompt(row:dict,text:str)->str:
    return (
        f"Read this publicly attributed {row['genre']} passage and answer only "
        "using its content. Excerpt:\n"
        f"{text}\n"
        f"Excerpt ID: {row['doc_id']}\n"
        "Question: What topic or idea does this excerpt primarily discuss?"
    )


def count(tokenizer,base:str,instruction:str)->int:
    prompt=base+"\nTask id: 0000. "+instruction
    return len(tokenizer.apply_chat_template(
        [{"role":"user","content":prompt}],
        tokenize=True,add_generation_prompt=True
    ))


def make(row:dict,tokenizer)->dict:
    target=row["target_prompt_tokens"]
    full=row["text"]
    if len(full)<7000:
        raise ValueError("insufficient pinned source")
    def prompt(chars:int):
        return body_prompt(row,full[:chars])
    if count(tokenizer,prompt(1),INSTRUCTIONS["label"])>target:
        raise ValueError("model prompt wrapper exceeds budget")
    if count(tokenizer,prompt(len(full)),INSTRUCTIONS["label"])<target-3:
        raise ValueError("actual source is shorter than desired prompt budget")
    lo,hi=1,len(full)
    while lo<hi:
        mid=(lo+hi)//2
        if count(tokenizer,prompt(mid),INSTRUCTIONS["label"])<target:
            lo=mid+1
        else:
            hi=mid
    text=prompt(lo)
    tokens={name:count(tokenizer,text,inst) for name,inst in INSTRUCTIONS.items()}
    if abs(tokens["label"]-target)>4:
        raise ValueError(f"tokenizer calibration mismatch {row['doc_id']}: {tokens}")
    if any(abs(v-target)>95 for v in tokens.values()):
        raise ValueError(f"output instruction changed input budget excessively: {tokens}")
    return {
        "request_id":row["doc_id"],
        "language":"en",
        "topic":row["genre"],
        "bucket":row["bucket"],
        "target_prompt_tokens":target,
        "prompt_text":text,
        "prompt_sha256":digest(text.encode("utf-8")),
        "source_excerpt_sha256":row["text_sha256"],
        "source_ebook_id":row["source_ebook_id"],
        "source_license": "US public domain as catalogued by Project Gutenberg",
        "calibrated_prompt_tokens":tokens,
        "source_characters_selected":lo,
    }


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("data/stage7_public_domain_passages.jsonl"))
    parser.add_argument("--source-manifest",type=Path,default=Path("data/stage7_public_domain_manifest.json"))
    parser.add_argument("--output",type=Path,default=Path("data/stage12_literary_requests.jsonl"))
    parser.add_argument("--manifest",type=Path,default=Path("data/stage12_literary_requests_manifest.json"))
    parser.add_argument("--verify-existing",action="store_true")
    a=parser.parse_args(argv)
    import os
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    from transformers import AutoTokenizer
    old=json.loads(a.source_manifest.read_text(encoding="utf-8"))
    original=a.source.read_bytes()
    if digest(original)!=old["corpus_jsonl_sha256"]:
        raise ValueError("Stage7 frozen source does not match manifest")
    rows=[json.loads(line) for line in original.decode("utf-8").splitlines()]
    cache=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
    matches=list(cache.glob("*"))
    if len(matches)!=1 or matches[0].name!=MODEL_SNAPSHOT:
        raise ValueError("expected exact cached Qwen model tokenizer")
    tok=AutoTokenizer.from_pretrained(str(matches[0]),local_files_only=True,trust_remote_code=False)
    out=[make(x,tok) for x in assign(rows)]
    content="".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in out)
    manifest={
        "schema":"stage12-independent-domain-natural-prose-v1",
        "corpus_origin":"Stage7 30 pinned Project Gutenberg excerpts",
        "original_corpus_sha256":old["corpus_jsonl_sha256"],
        "rights":"Source books catalogued as public domain in the United States; consult local copyright rules outside US",
        "source_project_attribution":"Project Gutenberg and original authors; no affiliation",
        "source_domain":"English literary and natural science prose, not Kubernetes documentation",
        "request_count":24,
        "genre_counts":dict(Counter(x["topic"] for x in out)),
        "language_counts":dict(Counter(x["language"] for x in out)),
        "bucket_counts":dict(Counter(x["bucket"] for x in out)),
        "tokenizer_snapshot":MODEL_SNAPSHOT,
        "target_prompt_tokens":TARGETS,
        "generation_instructions":INSTRUCTIONS,
        "request_jsonl_sha256":digest(content.encode("utf-8")),
        "source_files_not_refetched":True,
        "selected_ids":[x["request_id"] for x in out],
        "not_training_data":True,
    }
    serialized=json.dumps(manifest,indent=2,ensure_ascii=False)+"\n"
    if a.verify_existing:
        if a.output.read_text(encoding="utf-8")!=content or a.manifest.read_text(encoding="utf-8")!=serialized:
            raise ValueError("frozen non-Kubernetes held-out input changed")
        print("STAGE12_FROZEN_VALIDATED",len(out),manifest["request_jsonl_sha256"])
    else:
        if a.output.exists() or a.manifest.exists():
            raise FileExistsError("refuse to overwrite frozen independent domain inputs")
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(content,encoding="utf-8")
        a.manifest.write_text(serialized,encoding="utf-8")
        print("STAGE12_FROZEN_CREATED",len(out),manifest["request_jsonl_sha256"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
