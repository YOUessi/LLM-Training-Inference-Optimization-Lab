"""CPU-only validation before each independently sourced Stage12 GPU seed.

GitHub Actions can run this every 10 minutes; it cannot reach Tang GPU.
The pinned local Tang worker separately gates and executes approved jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FROZEN_SHA="fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04"


def preflight(root:Path)->dict:
    data=root/"data/stage12_literary_requests.jsonl"
    manifest=root/"data/stage12_literary_requests_manifest.json"
    raw=data.read_bytes()
    m=json.loads(manifest.read_text())
    if m["schema"]!="stage12-independent-domain-natural-prose-v1":
        raise ValueError("unexpected independent-domain corpus version")
    if hashlib.sha256(raw).hexdigest()!=FROZEN_SHA or m["request_jsonl_sha256"]!=FROZEN_SHA:
        raise ValueError("independent-domain request SHA mismatch")
    rows=[json.loads(x) for x in raw.decode("utf-8").splitlines()]
    if len(rows)!=24 or len({x["request_id"] for x in rows})!=24:
        raise ValueError("incorrect frozen cross-domain count")
    if len({x["topic"] for x in rows})!=5 or {x["language"] for x in rows}!={"en"}:
        raise ValueError("frozen corpus genre/language mismatch")
    if any(abs(y-x["target_prompt_tokens"])>95
           for x in rows for y in x["calibrated_prompt_tokens"].values()):
        raise ValueError("Qwen-tokenizer-calibrated length gate invalid")
    previous=root/"docs/evidence/stage11_heldout/stage11_analysis.json"
    result=json.loads(previous.read_text(encoding="utf-8"))
    if result["status"]!="pass" or result["offered"]!=288:
        raise ValueError("original-domain held-out prerequisite missing")
    return {
        "schema":"stage12-cross-domain-approved-cpu-preflight-v1",
        "status":"pass",
        "source_domain":"English classic literature/natural science, NOT Kubernetes",
        "source_license_scope":"public domain as catalogued in USA; local use requires checking jurisdiction",
        "new_source_sha256":FROZEN_SHA,
        "source_document_count":24,
        "genres":5,
        "tokenizer_snapshot":m["tokenizer_snapshot"],
        "prior_stage11_offered":result["offered"],
        "approved_new_seeds":[4093,5087],
        "rate":8,
        "deadlines_ms":[150,400],
        "control":"unlimited",
        "expected_stage12_offered":288,
        "planned_jobs":["stage12_seed_4093","stage12_seed_5087","stage12_crossdomain_analysis"],
        "git_hosted_runner_gpu_access":False,
    }


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    result=preflight(a.repo_root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print("STAGE12_CROSSDOMAIN_CPU_PREFLIGHT_PASS",result["expected_stage12_offered"])
    return 0


if __name__=="__main__":
    raise SystemExit(main())
