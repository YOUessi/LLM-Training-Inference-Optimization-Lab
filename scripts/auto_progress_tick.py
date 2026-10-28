"""Safely execute the next APPROVED Stage11 job from Tang user crontab.

Runs once per invocation (at most one job). Does NOT pull/execute mutable
GitHub main, does NOT install packages, and NEVER claims ChatGPT is running
in the background. It only executes this checked-in, pinned code revision.

Cron tick:
- lock against concurrent runs
- reject git HEAD drift and dirty tracked worktree
- run CPU tests and a fresh roadmap audit
- if GPU idle, run the next finite approved held-out or domain-shift seed
- then CPU analyze the completed pair
- preserve all raw errors/results; stop, never auto-retry a failed job
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

JOBS = (
    {"id":"heldout_seed_2027","requires_gpu":True,
     "argv":("bash","scripts/run_stage11_heldout_seed.sh","2027")},
    {"id":"heldout_seed_3031","requires_gpu":True,
     "argv":("bash","scripts/run_stage11_heldout_seed.sh","3031")},
    {"id":"stage11_heldout_analysis","requires_gpu":False,
     "argv":("python3","scripts/stage11_heldout_analyze.py")},
    {"id":"stage12_seed_4093","requires_gpu":True,
     "argv":("bash","scripts/run_stage12_heldout_seed.sh","4093")},
    {"id":"stage12_seed_5087","requires_gpu":True,
     "argv":("bash","scripts/run_stage12_heldout_seed.sh","5087")},
    {"id":"stage12_crossdomain_analysis","requires_gpu":False,
     "argv":("python3","scripts/stage12_crossdomain_analyze.py")},
    {"id":"stage13_k8s_seed6101","requires_gpu":True,
     "argv":("bash","scripts/run_stage13_mixed.sh","k8s","6101")},
    {"id":"stage13_k8s_seed6109","requires_gpu":True,
     "argv":("bash","scripts/run_stage13_mixed.sh","k8s","6109")},
    {"id":"stage13_prose_seed7201","requires_gpu":True,
     "argv":("bash","scripts/run_stage13_mixed.sh","prose","7201")},
    {"id":"stage13_prose_seed7209","requires_gpu":True,
     "argv":("bash","scripts/run_stage13_mixed.sh","prose","7209")},
    {"id":"stage13_all_analysis","requires_gpu":False,
     "argv":("bash","scripts/run_stage13_analysis.sh")},
)
SCHEMA = "llm-opt-tang-autoprogress-v1"


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(args: list[str], *, cwd: Path, env: dict | None = None,
        timeout: int = 25) -> subprocess.CompletedProcess:
    return subprocess.run(args,cwd=cwd,env=env,text=True,
                          capture_output=True,timeout=timeout,check=False)


def is_gpu_idle(root: Path) -> tuple[bool,str]:
    if shutil.which("nvidia-smi") is None:
        return False,"nvidia-smi unavailable"
    p=run(["nvidia-smi","--query-compute-apps=pid",
           "--format=csv,noheader"],cwd=root,timeout=10)
    if p.returncode!=0:
        return False,"cannot read NVIDIA compute owners"
    if any(line.strip() for line in p.stdout.splitlines()):
        return False,"other NVIDIA compute processes detected"
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("127.0.0.1",8026))==0:
            return False,"loopback vLLM port 8026 is occupied"
    return True,"GPU idle and port clear"


def json_atomic(path: Path, data: dict) -> None:
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def verify_completed_job(root:Path, jobid:str) -> None:
    if jobid.startswith("heldout_seed_"):
        seed=int(jobid.split("_")[-1])
        directory=root/"results"/f"stage11_seed_{seed}"
        complete=0
        for mode in ("label","detailed"):
            for policy,deadline in (("unlimited",250),("bounded",150),("bounded",400)):
                r=json.loads((directory/f"{mode}_{policy}_d{deadline}_seed{seed}.json").read_text())
                s=r["summary"];cfg=r["config"]
                if (r["status"]!="measured" or s["offered"]!=24 or
                    s["completed"]+s["rejected"]!=24 or s["errors"]!=0 or
                    cfg["seed"]!=seed or cfg["admission_deadline_ms"]!=deadline or
                    cfg["policy"]!=policy or cfg["mode"]!=mode or
                    not r["server_queue_monitor"]["available"] or
                    r["accepted_output_length_gate"]["status"]!="pass"):
                    raise ValueError(f"invalid accepted evidence: {mode}/{policy}/{deadline}/{seed}")
                complete+=s["offered"]
        if complete!=144:
            raise ValueError("held-out seed not fully measured")
    elif jobid.startswith("stage12_seed_"):
        seed=int(jobid.split("_")[-1])
        if seed not in (4093,5087):
            raise ValueError("unapproved cross-domain seed")
        directory=root/"results"/f"stage12_seed_{seed}"
        complete=0
        required_sha="fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04"
        for mode in ("label","detailed"):
            for policy,deadline in (("unlimited",250),("bounded",150),("bounded",400)):
                r=json.loads((directory/f"{mode}_{policy}_d{deadline}_seed{seed}.json").read_text())
                status=r["summary"];cfg=r["config"]
                if (r["status"]!="measured" or status["offered"]!=24 or
                    status["completed"]+status["rejected"]!=24 or status["errors"]!=0 or
                    cfg["seed"]!=seed or cfg["admission_deadline_ms"]!=deadline or
                    cfg["policy"]!=policy or cfg["mode"]!=mode or
                    cfg.get("frozen_requests_sha256")!=required_sha or
                    status["by_language"]["en"]["offered"]!=24 or
                    not r["server_queue_monitor"]["available"] or
                    r["accepted_output_length_gate"]["status"]!="pass"):
                    raise ValueError(f"unaccepted independent-domain case {seed} {mode}/{policy}/{deadline}")
                complete+=status["offered"]
        if complete!=144:
            raise ValueError("incomplete Stage12 independently sourced offers")
    elif jobid=="stage12_crossdomain_analysis":
        d=json.loads((root/"results/stage12_crossdomain_analysis.json").read_text())
        if (d["status"]!="pass" or d["offered"]!=288 or d["recorded_12_cases"]!=12 or
            set(d["heldout_seeds"])!={4093,5087} or
            d["frozen_domain_request_sha256"]!="fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04"):
            raise ValueError("Stage12 crossdomain analysis not accepted")
    elif jobid.startswith("stage13_") and jobid!="stage13_all_analysis":
        # Cron runs /usr/bin/python3 scripts/auto_progress_tick.py, where
        # sys.path[0] points to scripts/ rather than our src/ layout.
        # Resolve *only* the approved repository src path, never cwd/site
        # downloads or dynamic remote code.
        source_path=str((root/"src").resolve())
        if source_path not in sys.path:
            sys.path.insert(0,source_path)
        from llm_opt_lab.serving.stage10_admission import accounting
        accepted_pairs={
            ("k8s",6101),("k8s",6109),("prose",7201),("prose",7209),
        }
        field=jobid.split("_")
        if len(field)!=3 or not field[2].startswith("seed"):
            raise ValueError("invalid Stage13 job ID")
        domain,seed=field[1],int(field[2].removeprefix("seed"))
        if (domain,seed) not in accepted_pairs:
            raise ValueError("unapproved Stage13 domain/arrival seed")
        expected_hash={
            "k8s":"c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c",
            "prose":"fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04",
        }[domain]
        directory=root/"results"/f"stage13_{domain}_seed{seed}"
        commit=run(["git","rev-parse","HEAD"],cwd=root).stdout.strip()
        total=0
        for rate in (8,16):
            for policy in ("unlimited","fixed150","fixed400","adaptive"):
                p=directory/f"rate{rate}_{policy}_seed{seed}.json"
                data=json.loads(p.read_text())
                cfg=data["config"];summary=data["summary"]
                records=data["requests"]
                if (data["status"]!="measured" or
                    data["environment"]["git_commit"]!=commit or
                    cfg["frozen_requests_sha256"]!=expected_hash or
                    (cfg["domain"],cfg["seed"],cfg["rate_per_second"],cfg["policy"])!=(domain,seed,rate,policy) or
                    len(records)!=24 or summary["offered"]!=24 or summary["errors"]!=0 or
                    summary["completed"]+summary["rejected"]!=24 or
                    summary!=accounting(records,data["elapsed_s"]) or
                    data["actual_mode_summaries"]["label"]["offered"]!=12 or
                    data["actual_mode_summaries"]["detailed"]["offered"]!=12 or
                    not data["server_queue_monitor"]["available"] or
                    not data["kv_cache_monitor"]["available"] or
                    data["gate_final_state"]["active"]!=0 or
                    data["gate_final_state"]["waiters"]!=0):
                    raise ValueError(f"invalid Stage13 real vLLM case: {p}")
                total+=24
        if total!=192:
            raise ValueError("Stage13 seed missing valid offered requests")
    elif jobid=="stage13_all_analysis":
        d=json.loads((root/"results/stage13_full_analysis.json").read_text())
        expected_git=run(["git","rev-parse","HEAD"],cwd=root).stdout.strip()
        if (d.get("status")!="pass" or d.get("source_commit")!=expected_git or
            d.get("replayed_offered_requests")!=768 or d.get("real_case_count")!=32 or
            len(d.get("paired_scenarios",[]))!=8):
            raise ValueError("Stage13 768 real offered requests incomplete/invalid")
    elif jobid=="stage11_heldout_analysis":
        d=json.loads((root/"results/stage11_heldout_analysis.json").read_text())
        if (d["status"]!="pass" or d["offered"]!=288 or
            d["recorded_12_cases"]!=12 or
            set(d["heldout_seeds"])!={2027,3031}):
            raise ValueError("analysis of held-out runs not accepted")
    else:
        raise ValueError("unknown approved job")


def tick(root:Path, *,dry_run:bool=False) -> dict:
    repo=root.resolve()
    state_dir=repo/"results/auto_progress"
    state_dir.mkdir(parents=True,exist_ok=True)
    auth_file=state_dir/"approved_head.txt"
    if not auth_file.exists():
        raise RuntimeError("APPROVAL_REQUIRED: approved_head.txt missing")
    approved=auth_file.read_text().strip()
    if len(approved)!=40 or any(x not in "0123456789abcdef" for x in approved):
        raise RuntimeError("APPROVAL_INVALID: expected immutable full Git SHA")
    head=run(["git","rev-parse","HEAD"],cwd=repo)
    if head.returncode!=0 or head.stdout.strip()!=approved:
        raise RuntimeError("SOURCE_DRIFT: checked-out Git SHA differs from approval")
    dirty=run(["git","status","--porcelain","--untracked-files=normal"],cwd=repo)
    if dirty.returncode!=0 or dirty.stdout.strip():
        raise RuntimeError("DIRTY_WORKTREE: refusing unattended execution")

    state_file=state_dir/"state.json"
    state=json.loads(state_file.read_text()) if state_file.exists() else {
        "schema":SCHEMA, "source_sha":approved, "jobs":{},
    }
    if state.get("schema")!=SCHEMA or state.get("source_sha")!=approved:
        raise RuntimeError("STATE_MISMATCH: existing state from different Git SHA")

    pending=next((job for job in JOBS if state["jobs"].get(job["id"],{}).get("status")!="passed"),None)
    if pending is None:
        message="all preapproved jobs complete: no new unapproved work"
        if not dry_run:
            state["last_tick_utc"]=utcnow()
            state["last_observation"]=message
            json_atomic(state_file,state)
        return {"status":"complete","message":message,"source_sha":approved}

    prior=state["jobs"].get(pending["id"],{})
    if prior.get("status") in ("failed","running"):
        return {"status":"blocked_previous_failure_or_interruption",
                "job":pending["id"],
                "previous_status":prior["status"],
                "message":"manual inspection required; no silent retries"}
    gpu_idle,reason=is_gpu_idle(repo) if pending["requires_gpu"] else (True,"CPU-only analysis")
    if dry_run:
        return {"status":"dry_run","job":pending["id"],
                "gpu_available":gpu_idle,"gpu_reason":reason,"source_sha":approved}

    env=os.environ.copy()
    env["PYTHONPATH"]=str(repo/"src")
    env["HF_HUB_OFFLINE"]="1"
    env["TRANSFORMERS_OFFLINE"]="1"
    env["NO_PROXY"]="localhost,127.0.0.1"
    env["no_proxy"]=env["NO_PROXY"]
    env["PATH"]="/usr/local/bin:/usr/bin:/bin:"+env.get("PATH","")
    test=run(["python3","-m","unittest","discover","-s","tests","-q"],cwd=repo,env=env,timeout=60)
    state["last_tick_utc"]=utcnow()
    if test.returncode!=0:
        state["last_observation"]="CPU_TEST_FAILED, GPU job not started"
        (state_dir/"last_test_failure.log").write_text(test.stdout+"\n"+test.stderr)
        json_atomic(state_file,state)
        return {"status":"blocked_cpu_regression","job":pending["id"]}
    watcher=run(["python3","scripts/plan_watch.py","--out",str(state_dir/"latest_plan.json")],
                cwd=repo,env=env,timeout=20)
    if watcher.returncode!=0:
        state["last_observation"]="ROADMAP_AUDIT_FAILED"
        json_atomic(state_file,state)
        return {"status":"blocked_roadmap_audit","job":pending["id"]}
    if not gpu_idle:
        state["last_observation"]="GPU_SKIP: "+reason
        json_atomic(state_file,state)
        return {"status":"skipped_gpu_busy","job":pending["id"],"reason":reason}

    state["jobs"][pending["id"]]={"status":"running","started_utc":utcnow()}
    state["last_observation"]="running "+pending["id"]
    json_atomic(state_file,state)
    log=state_dir/(pending["id"]+".log")
    # POSIX timeout sends SIGTERM so the script's trap can shut down only
    # the owned vLLM process. A bounded kill-after avoids orphaned tests.
    argv=["timeout","--signal=TERM","--kill-after=20s","600s",*pending["argv"]]
    started=time.monotonic()
    try:
        completed=run(argv,cwd=repo,env=env,timeout=650)
        log.write_text(completed.stdout+"\n=== STDERR ===\n"+completed.stderr,encoding="utf-8")
        if completed.returncode!=0:
            raise RuntimeError(f"approved_job_exit={completed.returncode}")
        verify_completed_job(repo,pending["id"])
    except Exception as e:
        state["jobs"][pending["id"]]={"status":"failed","finished_utc":utcnow(),
                                     "reason":str(e)}
        state["last_observation"]="FAILED "+pending["id"]+": "+str(e)
        json_atomic(state_file,state)
        return {"status":"failed","job":pending["id"],"reason":str(e)}
    state["jobs"][pending["id"]]={"status":"passed","started_utc":
                                 state["jobs"][pending["id"]]["started_utc"],
                                 "finished_utc":utcnow(),
                                 "duration_seconds":round(time.monotonic()-started,2),
                                 "log_path":str(log.relative_to(repo))}
    state["last_observation"]="PASSED "+pending["id"]
    json_atomic(state_file,state)
    return {"status":"passed","job":pending["id"],
            "duration_seconds":round(time.monotonic()-started,2),
            "next_job":next((j["id"] for j in JOBS if
                            state["jobs"].get(j["id"],{}).get("status")!="passed"),None)}


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(argv)
    lockpath=a.repo_root/"results/auto_progress/tick.lock"
    lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("AUTO_PROGRESS",json.dumps({"status":"skipped_previous_tick_running"}))
            return 0
        try:
            result=tick(a.repo_root,dry_run=a.dry_run)
        except Exception as e:
            result={"status":"blocked","reason":str(e)}
        print("AUTO_PROGRESS",json.dumps(result,ensure_ascii=False),flush=True)
        return 0 if result["status"] in (
            "passed","complete","dry_run","skipped_gpu_busy",
            "skipped_previous_tick_running","blocked_cpu_regression",
            "blocked_roadmap_audit","blocked_previous_failure_or_interruption"
        ) else 2


if __name__=="__main__":
    raise SystemExit(main())
