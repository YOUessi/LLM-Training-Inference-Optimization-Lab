#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out="results/stage8_openloop_20261028"
if [[ $# -ge 1 ]]; then out="$1"; fi
if [[ -e "$out" ]]; then echo "BLOCKED: output already exists" >&2; exit 3; fi
if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: port 8026 occupied" >&2; exit 3; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "BLOCKED: another GPU process active" >&2; exit 3
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
  echo "STAGE8_CLEANUP status=$status"
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
    ready=1; break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "BLOCKED: vLLM startup failed" >&2
    tail -30 "$out/server.log" >&2; exit 4
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "BLOCKED: server startup timeout" >&2
  tail -30 "$out/server.log" >&2
  exit 4
fi
echo REAL_VLLM_READY
curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/models.json"
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_before.prom"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$out/gpu_loaded.txt"

for seed in 42 137; do
  if [[ "$seed" -eq 42 ]]; then
    mode_order='label summary detailed'
    rate_order='2 8'
  else
    mode_order='detailed summary label'
    rate_order='8 2'
  fi
  for mode in $mode_order; do
    for rate in $rate_order; do
      label="$(printf '%s_rate%s_seed%s' "$mode" "$rate" "$seed")"
      file="$out/$label.json"
      echo "REAL_ARRIVAL_CASE mode=$mode offered_rate=$rate seed=$seed"
      curl -fsS --max-time 4 http://127.0.0.1:8026/metrics > "$out/$label-metrics-before.prom"
      timeout 160 python3 -m llm_opt_lab.serving.stage8_arrivals \
        --model qwen2.5-1.5b-instruct-local \
        --mode "$mode" --rate "$rate" --seed "$seed" \
        --warmup 1 --output "$file"
      curl -fsS --max-time 4 http://127.0.0.1:8026/metrics > "$out/$label-metrics-after.prom"
    done
  done
done
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_after.prom"
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
n=0
for mode in ("label","summary","detailed"):
 for rate in (2,8):
  for seed in (42,137):
   f=root/f"{mode}_rate{rate}_seed{seed}.json"
   d=json.loads(f.read_text())
   assert d["status"]=="measured" and d["kind"]=="real_vllm_stage8_multilingual_openloop"
   assert d["summary"]["succeeded"]==24 and d["summary"]["failed"]==0
   assert d["summary"]["prompt_usage_coverage"]==1. and d["summary"]["usage_coverage"]==1.
   assert d["all_actual_input_tokens_in_bounds"] and d["length_gates"]["status"]=="pass"
   assert all(q["sent"]==8 and q["succeeded"]==8 for q in d["by_language"].values())
   n+=24
   print(mode,rate,seed,"p95_arrival_ttft",round(d["p95_ttft_from_scheduled_arrival_ms"],1),
         "p95_service_ttft",round(d["summary"]["p95_ttft_ms"],1),
         "real_out_mean",round(d["length_gates"]["realized_mean"],1),
         "output_tok_s",round(d["summary"]["output_tokens_per_second"],1))
print("STAGE8_REQUESTS",n)
assert n==288
PY
echo STAGE8_REAL_GPU_ALL_ACCEPTED
