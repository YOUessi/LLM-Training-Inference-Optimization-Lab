#!/usr/bin/env bash
set -euo pipefail

# End-to-end, real vLLM benchmark; loopback only; owned server only.
cd "$(dirname "$0")/.."
out="results/vllm_stage3_repeated_20261028"
if [[ $# -ge 1 ]]; then out="$1"; fi
if [[ -e "$out" ]]; then echo "BLOCKED: output exists: $out" >&2; exit 3; fi
if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: port 8026 busy" >&2; exit 3; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "BLOCKED: another GPU compute process is running" >&2
  exit 3
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
  echo "BENCHMARK_CLEANUP_COMPLETE exit_status=$status"
  exit "$status"
}
trap cleanup EXIT INT TERM
echo "SERVER_SETUP_START $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader > "$out/gpu_before.txt"
bash scripts/run_vllm_local.sh > "$out/server.log" 2>&1 &
server_pid=$!
echo "Controlled server PID=$server_pid"
ready=0
for _i in $(seq 1 120); do
  if curl -fsS --max-time 3 http://127.0.0.1:8026/health >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "BLOCKED: vLLM exited before ready" >&2
    tail -25 "$out/server.log" >&2
    exit 4
  fi
  sleep 1
done
if [[ "$ready" != 1 ]]; then
  echo "BLOCKED: startup timeout" >&2
  tail -25 "$out/server.log" >&2
  exit 4
fi
echo "VLLM_READY"
curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/models.json"
python3 - "$out/models.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
ids={x.get("id") for x in d.get("data",[])}
assert "qwen2.5-1.5b-instruct-local" in ids,ids
PY
nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader > "$out/gpu_loaded.txt"
run_sweep() {
  local label="$1" order="$2" requests="$3" max_tokens="$4" seed="$5"
  echo "SWEEP: $label concurrency=$order requests=$requests tokens=$max_tokens seed=$seed"
  local extra=()
  if [[ "$label" == "long_decode" ]]; then
    # Legacy long_decode reached 21 tokens despite cap=128; fail if short.
    extra=(--response-instruction
      "Provide a detailed explanation with at least 25 distinct numbered technical considerations. Do not stop after one sentence. Continue until the output budget is exhausted."
      --min-average-completion-tokens 90)
  fi
  timeout 240 python3 -m llm_opt_lab.serving.sweep \
    --engine-label vllm --base-url http://127.0.0.1:8026 \
    --model qwen2.5-1.5b-instruct-local \
    --requests "$requests" --warmup-requests 2 \
    --concurrency "$order" --max-tokens "$max_tokens" \
    --seed "$seed" --output-dir "$out/$label" \
    "${extra[@]}"
}
run_sweep short_forward '1,2,4,8' 24 32 42
run_sweep short_reverse '8,4,2,1' 24 32 137
run_sweep long_decode '1,4,8' 12 128 42
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
groups={}
for name in ("short_forward","short_reverse","long_decode"):
    d=json.loads((p/name/"summary.json").read_text())
    assert d["status"]=="measured" and d["engine_label"]=="vllm"
    groups[name]=[]
    for c in d["cases"]:
        s=c["summary"]
        assert s["succeeded"]==s["sent"] and s["failed"]==0
        assert s["usage_coverage"]==1.0
        groups[name].append({
            "concurrency":c["concurrency"],
            "p95_ttft_ms":s["p95_ttft_ms"],
            "p95_tpot_ms":s["p95_tpot_ms"],
            "output_tokens_per_second":s["output_tokens_per_second"],
            "succeeded":s["succeeded"],
        })
print(json.dumps(groups,indent=2))
PY
echo "ALL_SERVER_SWEEPS_COMPLETED"
