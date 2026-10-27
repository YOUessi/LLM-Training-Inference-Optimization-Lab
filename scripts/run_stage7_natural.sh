#!/usr/bin/env bash
set -euo pipefail

# Real vLLM non-repeating public-domain corpus + varied generation budgets.
# Loopback-only; no global system changes. Never overwrite results.
cd "$(dirname "$0")/.."
out="results/stage7_natural_20261028"
if [[ $# -gt 0 ]]; then out="$1"; fi
if [[ -e "$out" ]]; then echo "BLOCKED: output exists" >&2; exit 3; fi
if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: localhost 8026 occupied" >&2; exit 3; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "BLOCKED: another GPU task is active" >&2; exit 3
fi
mkdir -p "$out"
umask 077
export PYTHONPATH="$PWD/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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
 echo "STAGE7_SERVER_CLEANUP exit=$status"
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
    echo "BLOCKED: vLLM server exited before ready" >&2
    tail -30 "$out/server.log" >&2; exit 4
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then echo "BLOCKED: server readiness timeout" >&2;exit 4;fi
echo "SERVER_READY cache=off"
curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/models.json"
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_start.prom"
nvidia-smi --query-gpu=memory.used,utilization.gpu,driver_version --format=csv,noheader > "$out/gpu_loaded.txt"

for seed in 42 137; do
  if [[ "$seed" -eq 42 ]]; then order="1 2 4 8"; else order="8 4 2 1"; fi
  for mode in short long; do
    for concurrency in $order; do
      filename="$(printf '%s/%s_c%s_seed%s.json' "$out" "$mode" "$concurrency" "$seed")"
      echo "CASE genre=5 mode=$mode concurrent=$concurrency seed=$seed"
      timeout 200 python3 -m llm_opt_lab.serving.stage7_natural_bench \
        --model qwen2.5-1.5b-instruct-local \
        --mode "$mode" --concurrency "$concurrency" \
        --seed "$seed" --warmup 1 --output "$filename"
    done
  done
done
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_end.prom"
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
total=0
for seed in (42,137):
 for mode in ("short","long"):
  for c in (1,2,4,8):
   d=json.loads((root/f"{mode}_c{c}_seed{seed}.json").read_text())
   assert d["status"]=="measured" and d["config"]["selected_unique_passages"]==24
   assert d["summary"]["succeeded"]==24 and d["summary"]["failed"]==0
   assert d["summary"]["prompt_usage_coverage"]==1. and d["summary"]["usage_coverage"]==1.
   assert all(x["status"]=="pass" for x in d["bucket_statistics"].values())
   if mode=="long":assert d["generated_tokens_mean"]>=85
   total+=24
   print(seed,mode,c,"p95_TTFT",round(d["summary"]["p95_ttft_ms"],1),
         "output_tokens_s",round(d["summary"]["output_tokens_per_second"],1),
         "actual_output_tokens_mean",round(d["generated_tokens_mean"],1))
print("TOTAL_REAL_REQUESTS",total)
assert total==384
PY
echo "STAGE7_REAL_GPU_DATA_ALL_VALID"
