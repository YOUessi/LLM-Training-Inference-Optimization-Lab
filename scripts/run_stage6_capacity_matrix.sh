#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out="results/stage6_capacity_matrix_20261028"
if [[ $# -gt 0 ]]; then out="$1"; fi
if [[ -e "$out" ]]; then echo "BLOCKED: output exists" >&2; exit 3; fi
if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: port occupied" >&2; exit 3; fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
 echo "BLOCKED: another GPU process is running" >&2; exit 3
fi
mkdir -p "$out"
umask 077
export PYTHONPATH="$PWD/src"
export NO_PROXY=127.0.0.1,localhost
export no_proxy="$NO_PROXY"
export VLLM_PREFIX_CACHING=on
server_pid=""
cleanup() {
 status=$?
 trap - EXIT INT TERM
 if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
   kill -TERM "$server_pid" 2>/dev/null || :
   wait "$server_pid" 2>/dev/null || :
 fi
 echo "STAGE6_MATRIX_CLEANUP exit=$status"
 exit "$status"
}
trap cleanup EXIT INT TERM
echo "SOURCE_COMMIT $(git rev-parse HEAD)"

# Fresh server for each (capacity,filler) point: no cross-case KV reuse.
for spec in 128:4 256:4 512:4 512:12; do
 blocks="$(printf '%s' "$spec" | cut -d: -f1)"
 fillers="$(printf '%s' "$spec" | cut -d: -f2)"
 label="blocks$blocks-fillers$fillers"
 echo "=== KV CAPACITY POINT $label ==="
 if ss -ltn | grep -q ':8026 '; then echo "BLOCKED: port busy" >&2;exit 3;fi
 if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]';then
   echo "BLOCKED: other GPU process detected" >&2;exit 3
 fi
 mkdir -p "$out/$label"
 export VLLM_NUM_GPU_BLOCKS_OVERRIDE="$blocks"
 bash scripts/run_vllm_local.sh > "$out/$label/server.log" 2>&1 &
 server_pid=$!
 ready=0
 for n in $(seq 1 120); do
   if curl -fsS --max-time 3 http://127.0.0.1:8026/health >/dev/null 2>&1;then
     ready=1;break
   fi
   if ! kill -0 "$server_pid" 2>/dev/null;then
     echo "BLOCKED: capacity server crashed $label" >&2
     tail -20 "$out/$label/server.log" >&2
     exit 4
   fi
   sleep 1
 done
 if [[ "$ready" -ne 1 ]];then echo "BLOCKED: server readiness timeout $label" >&2;exit 4;fi
 curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/$label/models.json"
 echo "RUN CAPACITY $blocks FILLERS $fillers"
 timeout 120 python3 -m llm_opt_lab.serving.kv_capacity_curve \
   --model qwen2.5-1.5b-instruct-local \
   --blocks "$blocks" --fillers "$fillers" --seed 830 \
   --prefix-repetitions 35 --max-new-tokens 16 \
   --output "$out/$label/result.json"
 kill -TERM "$server_pid" || :
 wait "$server_pid" || :
 server_pid=""
 for n in $(seq 1 30);do
   if ! ss -ltn | grep -q ':8026 ';then break;fi
   sleep 1
 done
 if ss -ltn | grep -q ':8026 ';then echo "BLOCKED: port still occupied" >&2;exit 4;fi
done
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for blocks,fillers in ((128,4),(256,4),(512,4),(512,12)):
 path=root/f"blocks{blocks}-fillers{fillers}/result.json"
 d=json.loads(path.read_text())
 assert d["status"]=="measured",path
 assert d["config"]["blocks"]==blocks and d["config"]["fillers"]==fillers
 c=d["classification"]
 print(blocks,fillers,c["post_pressure_classification"],
       "hit_after",c["post_pressure_hit_fraction"])
print("STAGE6_MATRIX_COMPLETE")
PY
