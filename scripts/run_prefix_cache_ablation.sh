#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out="results/prefix_cache_ablation_20261028"
if [[ $# -ge 1 ]]; then out="$1"; fi
if [[ -e "$out" ]]; then echo "BLOCKED: output exists" >&2; exit 3; fi
if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: port busy" >&2; exit 3; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "BLOCKED: another GPU process is active" >&2; exit 3
fi
mkdir -p "$out"
umask 077
export PYTHONPATH="$PWD/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
server_pid=""
cleanup() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" 2>/dev/null || :
    wait "$server_pid" 2>/dev/null || :
  fi
  echo "CLEANUP_DONE status=$status"
  exit "$status"
}
trap cleanup EXIT INT TERM
echo "SOURCE_SHA $(git rev-parse HEAD)"
for state in off on; do
  echo "=== vLLM cache=$state ==="
  if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: port not freed" >&2; exit 3; fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
    echo "BLOCKED: GPU not free" >&2; exit 3
  fi
  mkdir -p "$out/$state"
  export VLLM_PREFIX_CACHING="$state"
  bash scripts/run_vllm_local.sh > "$out/$state/server.log" 2>&1 &
  server_pid=$!
  echo "OWNED_SERVER_PID=$server_pid"
  ready=0
  for n in $(seq 1 120); do
    if curl -fsS --max-time 3 http://127.0.0.1:8026/health >/dev/null 2>&1; then
      ready=1; break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "BLOCKED: crashed"; tail -20 "$out/$state/server.log"; exit 4
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then echo "BLOCKED: startup timeout" >&2; exit 4; fi
  echo "SERVER_READY cache=$state"
  curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/$state/models.json"
  curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/$state/metrics_start.prom"
  for seed in 42 137; do
    if [[ "$seed" == 42 ]]; then order='shared:1 distinct:1 shared:8 distinct:8'
    else order='distinct:8 shared:8 distinct:1 shared:1'; fi
    for spec in $order; do
      mode="$(printf '%s' "$spec" | cut -d: -f1)"
      concurrency="$(printf '%s' "$spec" | cut -d: -f2)"
      label="$mode-c$concurrency-seed$seed"
      echo "CASE cache=$state prefix=$mode concurrency=$concurrency seed=$seed"
      curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/$state/$label-before.prom"
      timeout 150 python3 -m llm_opt_lab.serving.prefix_cache_bench \
        --base-url http://127.0.0.1:8026 --model qwen2.5-1.5b-instruct-local \
        --declared-cache-state "$state" --mode "$mode" \
        --concurrency "$concurrency" --requests 24 --warmup-requests 2 \
        --prefix-repetitions 35 --min-prompt-tokens 900 --max-prompt-tokens 1450 \
        --max-new-tokens 32 --seed "$seed" --output "$out/$state/$label.json"
      curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/$state/$label-after.prom"
      # Enforce actual server prefix-token metrics, not only inferred TTFT.
      python3 -m llm_opt_lab.serving.prefix_cache_metrics \
        --before "$out/$state/$label-before.prom" \
        --after "$out/$state/$label-after.prom" \
        --cache-state "$state" --mode "$mode" \
        --output "$out/$state/$label-metrics.json"
    done
  done
  kill -TERM "$server_pid" || :
  wait "$server_pid" || :
  server_pid=""
  for n in $(seq 1 30); do
    if ! ss -ltn | grep -q ':8026 '; then break; fi
    sleep 1
  done
  if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: server not stopped"; exit 4; fi
  echo "SERVER_STOPPED cache=$state"
done
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); total=0
for state in ("off","on"):
    for seed in (42,137):
        for mode in ("shared","distinct"):
            for c in (1,8):
                d=json.loads((root/state/f"{mode}-c{c}-seed{seed}.json").read_text())
                s=d["summary"]
                assert d["status"]=="measured" and s["succeeded"]==24 and s["failed"]==0
                assert s["prompt_usage_coverage"]==1.0
                assert d["prompt_length_gate"]["all_lengths_within_bounds"]
                total+=24
                print(state,seed,mode,c,"mean_prompt",round(s["mean_prompt_tokens"],1),
                      "TTFT_P95",round(s["p95_ttft_ms"],2),
                      "out_tps",round(s["output_tokens_per_second"],2))
print("TOTAL_REAL_REQUESTS",total)
PY
echo PREFIX_CACHE_ABLATION_COMPLETED