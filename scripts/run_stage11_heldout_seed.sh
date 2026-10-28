#!/usr/bin/env bash
set -euo pipefail

# Finite GPU experiment for one held-out seed. Invoked by the safe cron runner.
cd "$(dirname "$0")/.."
seed="$1"
case "$seed" in 2027|3031) ;; *) echo "BLOCKED: seed not approved" >&2;exit 3;; esac
out="results/stage11_seed_$seed"
if [[ -e "$out" ]];then echo "BLOCKED: existing Stage11 result path" >&2;exit 3;fi
if ss -ltn | grep -q ':8026 ';then echo "BLOCKED: serving port used" >&2;exit 3;fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]';then
 echo "BLOCKED: GPU already has a compute process" >&2;exit 3
fi
mkdir -p "$out"
umask 077
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_PREFIX_CACHING=off
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE
server_pid=""
cleanup() {
 local status=$?
 trap - EXIT INT TERM
 if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null;then
  kill -TERM "$server_pid" 2>/dev/null || :
  wait "$server_pid" 2>/dev/null || :
 fi
 echo "STAGE11_CLEANUP status=$status"
 exit "$status"
}
trap cleanup EXIT INT TERM
echo "STAGE11_EXPERIMENT_GIT $(git rev-parse HEAD), seed=$seed"
nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader > "$out/gpu_before.txt"
bash scripts/run_vllm_local.sh > "$out/server.log" 2>&1 &
server_pid=$!
ready=0
for i in $(seq 1 120);do
 if curl -fsS --max-time 3 http://127.0.0.1:8026/health >/dev/null 2>&1;then
  ready=1;break
 fi
 if ! kill -0 "$server_pid" 2>/dev/null;then
  echo "BLOCKED: vLLM startup failed" >&2
  tail -22 "$out/server.log" >&2;exit 4
 fi
 sleep 1
done
if [[ "$ready" -ne 1 ]];then echo "BLOCKED: no model readiness" >&2;exit 4;fi
curl -fsS --max-time 6 http://127.0.0.1:8026/v1/models > "$out/models.json"
curl -fsS --max-time 6 http://127.0.0.1:8026/metrics > "$out/metrics_initial.prom"
if [[ "$seed" == 2027 ]];then
 spec_order="unlimited:250 bounded:150 bounded:400"
 mode_order="label detailed"
else
 spec_order="bounded:400 bounded:150 unlimited:250"
 mode_order="detailed label"
fi
for mode in $mode_order;do
 for spec in $spec_order;do
  policy="$(printf '%s' "$spec"|cut -d: -f1)"
  deadline="$(printf '%s' "$spec"|cut -d: -f2)"
  name="$(printf '%s_%s_d%s_seed%s' "$mode" "$policy" "$deadline" "$seed")"
  echo "STAGE11_REAL_CANDIDATE mode=$mode policy=$policy deadline_ms=$deadline seed=$seed"
  timeout 180 python3 -m llm_opt_lab.serving.stage10_admission \
   --model qwen2.5-1.5b-instruct-local \
   --mode "$mode" --policy "$policy" --seed "$seed" \
   --offered-rate 8 --deadline-ms "$deadline" \
   --output "$out/$name.json"
 done
done
python3 - "$out" "$seed" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);seed=int(sys.argv[2])
total=completed=rejected=0
for mode in ("label","detailed"):
 for policy,deadline in (("unlimited",250),("bounded",150),("bounded",400)):
  name=f"{mode}_{policy}_d{deadline}_seed{seed}.json"
  d=json.loads((root/name).read_text())
  c=d["config"];s=d["summary"]
  assert d["status"]=="measured" and s["offered"]==24
  assert s["accepted"]+s["rejected"]==24
  assert s["completed"]+s["rejected"]==24 and s["errors"]==0
  assert (c["seed"],c["policy"],c["mode"],c["admission_deadline_ms"])==(seed,policy,mode,deadline)
  assert c["fixed_offered_rate"]==8.
  assert d["accepted_output_length_gate"]["status"]=="pass"
  assert d["server_queue_monitor"]["available"]
  total+=24;completed+=s["completed"];rejected+=s["rejected"]
  print("CASE",mode,policy,deadline,
        "completed",s["completed"],"rejected",s["rejected"],
        "SLO250",round(s["slo_all_offered"]["250"]["compliance_fraction"],3),
        "useful_TPS",round(s["useful_output_tokens_per_second"],1))
assert total==144 and total==completed+rejected
print("STAGE11_SUCCESS",seed,"total",total,"completed",completed,"rejected",rejected)
PY
curl -fsS --max-time 6 http://127.0.0.1:8026/metrics > "$out/metrics_final.prom"
echo "STAGE11_SEED_FINISHED $seed"
