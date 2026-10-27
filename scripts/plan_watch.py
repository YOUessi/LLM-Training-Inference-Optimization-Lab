"""Half-hour GitHub Actions project plan audit (CPU-only, deterministic).

This script DOES NOT reach Tang or execute GPU experiments. It analyzes current
GitHub repository evidence vs roadmap and emits next verifiable action,
so the scheduled workflow can continuously run tests and compare progress
without inventing remote hardware results.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args(argv: list[str] | None = None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--summary", type=Path)
    return p.parse_args(argv)


def audit(root: Path) -> dict:
    plan = root / "docs" / "ROADMAP.md"
    if not plan.exists():
        raise FileNotFoundError(f"missing authoritative roadmap: {plan}")
    markdown=plan.read_text(encoding="utf-8")
    counts={"completed":len(re.findall(r"^\s*- \[x\]",markdown,re.M)),
            "open":len(re.findall(r"^\s*- \[ \]",markdown,re.M))}
    reports={
        "Stage 3 vLLM real SSE":(root/"docs/STAGE3_REPORT_2026-10-28.md").exists(),
        "Stage 4 Prefix Cache":(root/"docs/STAGE4_PREFIX_CACHE_REPORT_2026-10-28.md").exists(),
        "Stage 5 KV capacity":(root/"docs/STAGE5_KV_CAPACITY_REPORT_2026-10-28.md").exists(),
        "Stage 5 Mixed Context":(root/"docs/STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md").exists(),
        "Stage 6 SLO frontier":(root/"docs/STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md").exists(),
        "Stage 6 KV capacity matrix":(root/"docs/STAGE6_CAPACITY_MATRIX_REPORT_2026-10-28.md").exists(),
    }
    implementation={
        "KV capacity probe":(root/"src/llm_opt_lab/serving/kv_capacity_probe.py").exists(),
        "Mixed context probe":(root/"src/llm_opt_lab/serving/mixed_context_bench.py").exists(),
        "Stage5 eviction launcher":(root/"scripts/run_stage5_eviction.sh").exists(),
        "Stage5 mixed launcher":(root/"scripts/run_stage5_mixed.sh").exists(),
    }
    actions=[]
    if not all(implementation.values()):
        actions.append("Implement and test missing Stage5 CPU-side scripts.")
    if not reports["Stage 5 KV capacity"]:
        actions.append("After GPU preflight on Tang, run actual bounded KV cache eviction test, inspect Prometheus hit counters, then commit raw evidence and report.")
    if not reports["Stage 5 Mixed Context"]:
        actions.append("After GPU preflight on Tang, run heterogeneous-input real SSE workload at 1/2/4/8 concurrency and publish actual prompt tokens, TTFT/TPOT and throughput.")
    stage5_complete = (
        reports["Stage 5 KV capacity"] and
        reports["Stage 5 Mixed Context"]
    )
    if stage5_complete and not reports["Stage 6 SLO frontier"]:
        actions.append("Compute and publish Stage6 P95 TTFT SLO frontier from validated Stage5 real SSE evidence.")
    if stage5_complete and reports["Stage 6 SLO frontier"]:
        if not reports["Stage 6 KV capacity matrix"]:
            actions.append("Design and run Tang-GPU, multi-capacity 128/256/512 KV cache-pressure suite; requires authorized online Tang execution.")
        else:
            actions.append("Stage6 multi-capacity 128/256/512 KV retention has real evidence: next freeze a heterogeneous nonrepetitive corpus and output-length mixtures.")
        actions.append("Repeat Stage6 P95 TTFT SLO frontier on frozen diverse text with varied request arrival and output lengths; do not infer production SLO from two seeds.")
    actions.append("Keep Nsight Compute hardware counters and multi-GPU FSDP BLOCKED until authorized permissions/hardware exist.")
    try:
        x=subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,text=True,timeout=4,check=False)
        commit=x.stdout.strip() if x.returncode==0 else None
    except OSError:
        commit=None
    return {
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "git_commit":commit,
        "roadmap_path":"docs/ROADMAP.md",
        "roadmap_checkbox_counts":counts,
        "published_reports":reports,
        "stage5_implementation":implementation,
        "next_executable_actions":actions,
        "gpu_access_available_in_hosted_ci":False,
        "can_execute_remote_tang_from_hosted_ci":False,
        "status":"stage5_evidence_published" if stage5_complete else "stage5_evidence_pending",
    }


def main(argv: list[str] | None = None) -> int:
    a=parse_args(argv)
    data=audit(a.repo_root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if a.summary:
        a.summary.parent.mkdir(parents=True,exist_ok=True)
        lines=[
            "# LLM Optimization Lab — Scheduled Roadmap Audit",
            "",
            f"- Commit: `{data['git_commit']}`",
            f"- Status: **{data['status']}**",
            f"- Roadmap checkboxes: {data['roadmap_checkbox_counts']['completed']} done, "
            f"{data['roadmap_checkbox_counts']['open']} pending",
            "- **Important:** GitHub hosted runners are CPU-only and cannot access Tang's GPU or the user's live ChatGPT conversation.",
            "",
            "## Report presence",
            "",
        ]
        for title,present in data["published_reports"].items():
            lines.append(f"- {'[x]' if present else '[ ]'} {title}")
        lines+=["","## Implementation gates",""]
        for title,present in data["stage5_implementation"].items():
            lines.append(f"- {'[x]' if present else '[ ]'} {title}")
        lines+=["","## Next verifiable actions",""]
        for i,item in enumerate(data["next_executable_actions"],1):
            lines.append(f"{i}. {item}")
        a.summary.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("ROADMAP_AUDIT",data["git_commit"],data["status"],
          "open",data["roadmap_checkbox_counts"]["open"],flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
