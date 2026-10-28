#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

out="results/stage9_worker_isolation_20261028"
if [[ $# -gt 0 ]]; then out="$1"; fi
if [[ -e "$out" ]]; then echo "BLOCKED: existing results path" >&2; exit 3; fi
if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: service port already occupied" >&2;exit 3;fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "BLOCKED: other CUDA compute task running" >&2;exit 3
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
  code=$?
  trap - EXIT INT TERM
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || :
    wait "$server_pid" 2>/dev/null || :
  fi
  echo "STAGE9_SERVER_CLEANUP exit=$code"
  exit "$code"
}
trap cleanup EXIT INT TERM
echo "SOURCE_GIT_HEAD $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader > "$out/gpu_before.txt"
bash scripts/run_vllm_local.sh > "$out/server.log" 2>&1 &
server_pid=$!
ready=0
for i in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8026/health >/dev/null 2>&1; then
    ready=1;break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "BLOCKED: model server crashed" >&2;tail -30 "$out/server.log" >&2;exit 4
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then echo "BLOCKED: model server readiness timeout" >&2;exit 4;fi
echo "SERVER_READY"
curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/models.json"
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/server_metrics_start.prom"

for seed in 42 137; do
  if [[ "$seed" -eq 42 ]]; then
    worker_order="8 32"
    condition_order="label:8 detailed:2 detailed:8"
  else
    worker_order="32 8"
    condition_order="detailed:8 detailed:2 label:8"
  fi
  for workers in $worker_order; do
    for spec in $condition_order; do
      mode="$(printf '%s' "$spec" | cut -d: -f1)"
      rate="$(printf '%s' "$spec" | cut -d: -f2)"
      label="$(printf '%s_w%s_rate%s_seed%s' "$mode" "$workers" "$rate" "$seed")"
      echo "STAGE9 real=1 mode=$mode workers=$workers offered=$rate seed=$seed"
      curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/$label-before.prom"
      timeout 165 python3 -m llm_opt_lab.serving.stage8_arrivals \
        --model qwen2.5-1.5b-instruct-local \
        --mode "$mode" --rate "$rate" --seed "$seed" \
        --schedule-kind fixed_window --max-client-workers "$workers" \
        --warmup 1 --output "$out/$label.json"
      curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/$label-after.prom"
    done
  done
done
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/server_metrics_end.prom"
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
total=0
for seed in (42,137):
 for workers in (8,32):
  for mode,rate in (("label",8),("detailed",2),("detailed",8)):
   name=f"{mode}_w{workers}_rate{rate}_seed{seed}.json"
   d=json.loads((root/name).read_text())
   assert d["status"]=="measured" and d["summary"]["succeeded"]==24
   assert d["summary"]["failed"]==0 and d["all_actual_input_tokens_in_bounds"]
   assert d["config"]["max_client_workers"]==workers
   assert d["summary"]["prompt_usage_coverage"]==1.
   assert d["summary"]["usage_coverage"]==1.
   assert d["config"]["schedule_kind"]=="fixed_window"
   assert abs(d["offered_arrival_rate_empirical_per_second"]-rate)<1e-8
   total+=24
   print(name,"p95_sched_ttft",round(d["p95_ttft_from_scheduled_arrival_ms"],1),
         "p95_send_ttft",round(d["summary"]["p95_ttft_ms"],1),
         "p95_clientwait",round(d["p95_executor_queue_wait_ms"],1),
         "max_server_waiting",d["server_request_queue_gauges"]["max_server_waiting_requests"])
print("STAGE9_REAL_REQUESTS",total)
assert total==288
PY
echo "STAGE9_WORKER_ISOLATION_SUCCESS"
