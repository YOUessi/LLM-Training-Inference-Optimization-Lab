"""CPU-only scheduled next-step validation for frozen Stage11 held-out work.

GitHub Actions does NOT have access to Tang's GPU or localhost results.
This check verifies the pinned source corpus, Stage10 baseline evidence,
held-out seed/threshold plan and emits an executable task manifest for
a separately authorized Tang user-cron executor. Does not run remote code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def preflight(root:Path) -> dict:
    data=root/"data/stage8_frozen_requests.jsonl"
    manifest=root/"data/stage8_frozen_requests_manifest.json"
    m=json.loads(manifest.read_text(encoding="utf-8"))
    corpus=data.read_bytes()
    if hashlib.sha256(corpus).hexdigest()!=m["request_jsonl_sha256"]:
        raise ValueError("Stage11 workload corpus SHA drift")
    sources=[json.loads(x) for x in corpus.decode("utf-8").splitlines()]
    if len(sources)!=24 or len({x["request_id"] for x in sources})!=24:
        raise ValueError("not 24 unique frozen model prompts")
    if {x["language"] for x in sources}!={"en","zh-cn","ja"}:
        raise ValueError("not 3 licensed language groups")
    baseline=root/"docs/evidence/stage10_admission/stage10_analysis.json"
    prev=json.loads(baseline.read_text(encoding="utf-8"))
    if prev["status"]!="pass" or prev["total_offered_requests"]!=192:
        raise ValueError("Stage10 source of research question is incomplete")
    queue=[
        {"id":"heldout_seed_2027","requires_gpu":True,"seed":2027,"modes":["label","detailed"],"deadlines_ms":[150,400]},
        {"id":"heldout_seed_3031","requires_gpu":True,"seed":3031,"modes":["label","detailed"],"deadlines_ms":[150,400]},
        {"id":"stage11_heldout_analysis","requires_gpu":False,"depends_on":["heldout_seed_2027","heldout_seed_3031"]},
    ]
    return {
        "schema":"llm-stage11-approved-next-steps-v1",
        "status":"preflight_pass",
        "frozen_corpus_sha256":m["request_jsonl_sha256"],
        "previous_stage10_real_offered":192,
        "heldout_seed_set":[2027,3031],
        "fixed_offered_rate":8,
        "client_workers":32,
        "bounded_inflight":8,
        "bounded_queue_slots":4,
        "policy_deadlines_ms":[150,400],
        "expected_offered_requests":288,
        "approved_work_items":queue,
        "gpu_access_from_github_actions":False,
        "tang_execution_requires_pinned_sha_and_idle_gpu":True,
    }


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    data=preflight(a.repo_root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    print("STAGE11_CPU_PREFLIGHT_PASS",data["expected_offered_requests"],
          "preapproved offers; no remote GPU access in GitHub Actions")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
