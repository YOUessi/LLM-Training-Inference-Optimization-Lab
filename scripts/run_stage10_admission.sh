#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out="results/stage10_admission_20261028"
if [[ $# -gt 0 ]]; then out="$1"; fi
if [[ -e "$out" ]]; then echo "BLOCKED: results path exists" >&2; exit 3; fi
if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: local model port is in use" >&2; exit 3; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "BLOCKED: another GPU compute process is running" >&2;exit 3
fi

mkdir -p "$out"
umask 077
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export VLLM_PREFIX_CACHING=off
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE
server_pid=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || :
    wait "$server_pid" 2>/dev/null || :
  fi
  echo "STAGE10_CONTROLLED_SERVER_CLEANUP exit=$status"
  exit "$status"
}
trap cleanup EXIT INT TERM
echo "SOURCE_COMMIT $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader > "$out/gpu_before.txt"
bash scripts/run_vllm_local.sh > "$out/server.log" 2>&1 &
server_pid=$!
ready=0
for i in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8026/health >/dev/null 2>&1; then
    ready=1;break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "BLOCKED: model server exited" >&2
    tail -30 "$out/server.log" >&2
    exit 4
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "BLOCKED: model startup did not complete" >&2
  tail -30 "$out/server.log" >&2; exit 4
fi
echo REAL_VLLM_READY
curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/models.json"
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/server_metrics_start.prom"
for seed in 42 137; do
  if [[ "$seed" -eq 42 ]]; then
    policy_order="unlimited bounded"
    mode_order="label detailed"
  else
    policy_order="bounded unlimited"
    mode_order="detailed label"
  fi
  for policy in $policy_order; do
    for mode in $mode_order; do
      label="$(printf '%s_%s_seed%s' "$policy" "$mode" "$seed")"
      echo "STAGE10_REAL offered=24 rate=8 mode=$mode policy=$policy seed=$seed"
      curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/$label-before.prom"
      timeout 180 python3 -m llm_opt_lab.serving.stage10_admission \
        --model qwen2.5-1.5b-instruct-local \
        --mode "$mode" --policy "$policy" --seed "$seed" \
        --output "$out/$label.json"
      curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/$label-after.prom"
    done
  done
done
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/server_metrics_end.prom"
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
n=0
for mode in ("label","detailed"):
 for policy in ("unlimited","bounded"):
  for seed in (42,137):
   p=root/f"{policy}_{mode}_seed{seed}.json"
   d=json.loads(p.read_text())
   assert d["status"]=="measured"
   assert d["summary"]["offered"]==24
   assert d["summary"]["errors"]==0
   assert d["summary"]["accepted"]+d["summary"]["rejected"]==24
   assert d["summary"]["completed"]+d["summary"]["rejected"]==24
   assert d["config"]["policy"]==policy
   assert d["server_queue_monitor"]["available"]
   assert d["accepted_output_length_gate"]["status"]=="pass"
   assert all(x["status"] in ("ok","rejected") for x in d["requests"])
   n+=24
   print(mode,policy,seed,
         "completed",d["summary"]["completed"],
         "rejected",d["summary"]["rejected"],
         "all_offered_250ms_SLO",round(d["summary"]["slo_all_offered"]["250"]["compliance_fraction"],3),
         "useful_tps",round(d["summary"]["useful_output_tokens_per_second"],1))
print("TOTAL_OFFERED_REAL_REQUESTS",n)
assert n==192
PY
echo STAGE10_ADMISSION_REAL_TRIAL_COMPLETED
