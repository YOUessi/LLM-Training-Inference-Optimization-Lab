"""Freeze Qwen-token-calibrated multilingual serving requests, never train.

Runs on Tang (existing cached Qwen tokenizer), outputs one deterministic
24-request JSONL and provenance manifest. Must be committed to GitHub before
live benchmark execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

BUCKET_TOKENS = {
    "short": 170,
    "medium": 300,
    "long": 470,
    "very_long": 680,
}
LANGUAGES = ("en", "zh-cn", "ja")
MODEL_ID = "Qwen2.5-1.5B-Instruct"
MODEL_SNAPSHOT = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
# Keep these frozen before measuring natural output lengths.
INSTRUCTIONS = {
    "label": "Name the main Kubernetes resource category in a short noun phrase. No explanation.",
    "summary": "Summarize the technical content in one or two informative sentences.",
    "detailed": "Give at least 20 distinct numbered technical considerations from the excerpt. Continue until the output limit.",
}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def assign_docs(rows: list[dict]) -> list[dict]:
    if len(rows) != 24 or {r["language"] for r in rows} != set(LANGUAGES):
        raise ValueError("expected 24 excerpts spanning 3 languages")
    by_lang=defaultdict(list)
    for r in rows:
        by_lang[r["language"]].append(r)
    jobs=[]
    for lang in LANGUAGES:
        candidates=sorted(by_lang[lang],key=lambda r:r["doc_id"])
        if len(candidates)!=8:
            raise ValueError(f"expected 8 docs for {lang}")
        random.Random(2026 + LANGUAGES.index(lang)).shuffle(candidates)
        for i,doc in enumerate(candidates):
            b=list(BUCKET_TOKENS)[i//2]
            jobs.append({**doc,"bucket":b,"target_tokens":BUCKET_TOKENS[b]})
    if set(Counter(x["bucket"] for x in jobs).values()) != {6}:
        raise AssertionError("not 6 unique docs per bucket")
    if len({x["doc_id"] for x in jobs})!=24:
        raise ValueError("duplicate source excerpts in a run")
    return jobs


def build_prompt(doc: dict, excerpt: str) -> str:
    return (
        f"Read the Kubernetes technical document. Source language: {doc['language']}; "
        f"subject: {doc['topic']}. Use only the excerpt for the answer.\n"
        f"Excerpt:\n{excerpt}\n"
        "Question: What is the main resource or configuration behavior described?"
    )


def tok_count(tokenizer, prompt: str, instruction: str) -> int:
    user = prompt + "\nTask id: 0000. " + instruction
    ids=tokenizer.apply_chat_template(
        [{"role":"user","content":user}],tokenize=True,
        add_generation_prompt=True,
    )
    return len(ids)


def freeze_doc(doc: dict, tokenizer) -> dict:
    full=doc["text"]
    target=doc["target_tokens"]
    if len(full)<1800:
        raise ValueError("not enough excerpt content")
    smallest=build_prompt(doc,full[:1])
    if tok_count(tokenizer,smallest,INSTRUCTIONS["label"]) > target:
        raise ValueError("prompt wrapper exceeds target")
    if tok_count(tokenizer,build_prompt(doc,full),INSTRUCTIONS["label"]) < target-3:
        raise ValueError(f"source excerpt cannot reach token budget: {doc['doc_id']}")
    lo,hi=1,len(full)
    while lo<hi:
        mid=(lo+hi)//2
        size=tok_count(tokenizer,build_prompt(doc,full[:mid]),INSTRUCTIONS["label"])
        if size<target:
            lo=mid+1
        else:
            hi=mid
    prompt=build_prompt(doc,full[:lo])
    # A shared excerpt is used for all generation modes; input instructions
    # are necessarily different, so exact real token lengths are checked
    # separately per mode during online serving.
    counts={k:tok_count(tokenizer,prompt,v) for k,v in INSTRUCTIONS.items()}
    if not target-3 <= counts["label"] <= target+3:
        raise ValueError(f"cannot fit base token target for {doc['doc_id']}: {counts}")
    if any(abs(n-target)>95 for n in counts.values()):
        raise ValueError(f"instructions cross bucket bounds for {doc['doc_id']}: {counts}")
    return {
        "request_id": doc["doc_id"],
        "language": doc["language"],
        "topic": doc["topic"],
        "bucket": doc["bucket"],
        "target_prompt_tokens": target,
        "excerpt_source_sha256": doc["text_sha256"],
        "prompt_text": prompt,
        "prompt_sha256": sha(prompt.encode("utf-8")),
        "prefix_characters": lo,
        "calibrated_prompt_tokens": counts,
    }


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus",type=Path,default=Path("data/stage8_multilingual_k8s_docs.jsonl"))
    p.add_argument("--corpus-manifest",type=Path,default=Path("data/stage8_multilingual_k8s_manifest.json"))
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--output",type=Path,default=Path("data/stage8_frozen_requests.jsonl"))
    p.add_argument("--manifest",type=Path,default=Path("data/stage8_frozen_requests_manifest.json"))
    p.add_argument("--verify-existing",action="store_true")
    a=p.parse_args(argv)
    import os
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    from transformers import AutoTokenizer
    meta=json.loads(a.corpus_manifest.read_text(encoding="utf-8"))
    raw=a.corpus.read_bytes()
    if sha(raw)!=meta["corpus_jsonl_sha256"]:
        raise ValueError("parent frozen corpus changed")
    rows=[json.loads(v) for v in raw.decode("utf-8").splitlines()]
    for r in rows:
        if sha(r["text"].encode("utf-8"))!=r["text_sha256"]:
            raise ValueError("input excerpt hash mismatch")
    if a.model_dir is None:
        base=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
        matches=list(base.glob("*"))
        if len(matches)!=1:
            raise ValueError("expected one exact cached Qwen snapshot; pass --model-dir")
        a.model_dir=matches[0]
    if a.model_dir.name != MODEL_SNAPSHOT:
        raise ValueError("unexpected cached model snapshot")
    tokenizer=AutoTokenizer.from_pretrained(str(a.model_dir),local_files_only=True,
                                              trust_remote_code=False)
    jobs=assign_docs(rows)
    output=[freeze_doc(doc,tokenizer) for doc in jobs]
    lines="".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in output)
    manifest={
        "schema":"stage8-frozen-qwen-token-calibrated-v1",
        "original_corpus_sha256":meta["corpus_jsonl_sha256"],
        "tokenizer_snapshot":MODEL_SNAPSHOT,
        "tokenizer_id":MODEL_ID,
        "language_counts":dict(Counter(x["language"] for x in output)),
        "bucket_counts":dict(Counter(x["bucket"] for x in output)),
        "generation_instructions":INSTRUCTIONS,
        "target_prompt_tokens":BUCKET_TOKENS,
        "request_count":len(output),
        "request_jsonl_sha256":sha(lines.encode("utf-8")),
        "not_a_training_dataset":True,
        "selection":"fixed 2026 seed by source language and four token budgets",
    }
    serialized=json.dumps(manifest,indent=2,ensure_ascii=False)+"\n"
    if a.verify_existing:
        if a.output.read_text(encoding="utf-8")!=lines or a.manifest.read_text(encoding="utf-8")!=serialized:
            raise ValueError("pre-frozen request corpus unexpectedly changed")
        print("FROZEN_STAGE8_REQUESTS_VERIFIED",len(output),manifest["request_jsonl_sha256"])
        return 0
    if a.output.exists() or a.manifest.exists():
        raise FileExistsError("request manifest exists; use --verify-existing")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(lines,encoding="utf-8")
    a.manifest.write_text(serialized,encoding="utf-8")
    print("FROZEN_STAGE8_REQUESTS",len(output),manifest["request_jsonl_sha256"])
    for bucket in BUCKET_TOKENS:
        group=[x for x in output if x["bucket"]==bucket]
        print(bucket, "calibrated_label_range",
              min(x["calibrated_prompt_tokens"]["label"] for x in group),
              max(x["calibrated_prompt_tokens"]["label"] for x in group))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
