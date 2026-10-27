#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out="results/stage5_eviction_20261028"
if [[ $# -gt 0 ]]; then out="$1"; fi
if [[ -e "$out" ]]; then echo "BLOCKED: output already exists" >&2; exit 3; fi
if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: port 8026 occupied" >&2; exit 3; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "BLOCKED: other GPU compute process active" >&2; exit 3
fi
mkdir -p "$out"
umask 077
export PYTHONPATH="$PWD/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export VLLM_PREFIX_CACHING=on
export VLLM_NUM_GPU_BLOCKS_OVERRIDE=256
server_pid=""
cleanup() {
 status=$?
 trap - EXIT INT TERM
 if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
   kill -TERM "$server_pid" 2>/dev/null || :
   wait "$server_pid" 2>/dev/null || :
 fi
 echo "STAGE5_EVICTION_CLEANUP status=$status"
 exit "$status"
}
trap cleanup EXIT INT TERM
echo "Git HEAD $(git rev-parse HEAD)"
echo "Launching loopback-only capacity-constrained vLLM..."
bash scripts/run_vllm_local.sh > "$out/server.log" 2>&1 &
server_pid=$!
ready=0
for i in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8026/health >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "BLOCKED: vLLM server crashed" >&2
    tail -28 "$out/server.log" >&2
    exit 4
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "BLOCKED: server readiness timeout" >&2; tail -25 "$out/server.log" >&2; exit 4
fi
curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/models.json"
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_before.prom"
echo "RUNNING KV CAPACITY PRESSURE PROBE"
if timeout 210 python3 -m llm_opt_lab.serving.kv_capacity_probe \
  --model qwen2.5-1.5b-instruct-local --num-gpu-blocks-override 256 \
  --distinct-fillers 12 --prefix-repetitions 35 --seed 830 \
  --max-new-tokens 16 --output "$out/capacity.json"; then
  echo "EVICTION_PROBE_PASS"
else
  echo "EVICTION_PROBE_NOT_PASS: preserve genuine negative or blocker result" >&2
  exit 5
fi
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_after.prom"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$out/gpu_loaded.txt"
echo "EVICTION_COMPLETE"
