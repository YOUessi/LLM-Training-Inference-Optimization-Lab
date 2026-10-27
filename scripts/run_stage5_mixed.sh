#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out="results/stage5_mixed_20261028"
if [[ $# -gt 0 ]]; then out="$1"; fi
if [[ -e "$out" ]]; then echo "BLOCKED: output exists" >&2; exit 3; fi
if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: port busy" >&2; exit 3; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "BLOCKED: other GPU process active" >&2; exit 3
fi
mkdir -p "$out"
umask 077
export PYTHONPATH="$PWD/src"
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
 echo "STAGE5_MIXED_CLEANUP status=$status"
 exit "$status"
}
trap cleanup EXIT INT TERM
echo "Git HEAD $(git rev-parse HEAD)"
bash scripts/run_vllm_local.sh > "$out/server.log" 2>&1 &
server_pid=$!
ready=0
for i in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8026/health >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "BLOCKED: model did not load" >&2
    tail -25 "$out/server.log" >&2;exit 4
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then echo "BLOCKED: timeout";exit 4;fi
curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/models.json"
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_before.prom"
for concurrency in 1 2 4 8; do
  for seed in 42 137; do
    echo "MIXED_PROMPT concurrent=$concurrency seed=$seed"
    casefile="$(printf '%s/mixed_c%s_seed%s.json' "$out" "$concurrency" "$seed")"
    timeout 210 python3 -m llm_opt_lab.serving.mixed_context_bench \
      --model qwen2.5-1.5b-instruct-local \
      --concurrency "$concurrency" --seed "$seed" \
      --repeats-per-length 6 --max-new-tokens 32 --warmup 1 \
      --output "$casefile"
  done
done
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_after.prom"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > "$out/gpu_loaded.txt"
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
all_rows=[]
for concurrency in (1,2,4,8):
 for seed in (42,137):
  p=root/f"mixed_c{concurrency}_seed{seed}.json"
  d=json.loads(p.read_text())
  assert d["status"]=="measured" and d["summary"]["succeeded"]==24
  assert all(b["status"]=="pass" for b in d["by_prompt_length"].values())
  all_rows.append((concurrency,seed,d["summary"]["p95_ttft_ms"],
                   d["summary"]["output_tokens_per_second"]))
print("STAGE5 MIXED ALL 8 CASES PASS")
for row in all_rows:print(row)
PY
echo "MIXED_CONTEXT_COMPLETE"
