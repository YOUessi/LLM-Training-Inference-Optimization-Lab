#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage43_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_UNAPPROVED_SOURCE;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_SHARED_LOCK;exit 3; }
! ss -ltn|grep -q ':8033 ' || { echo BLOCKED_DUAL_MODEL_PORT_OCCUPIED;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_OTHER_VLLM_PORT;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_EXISTING_GPU_PROCESS;exit 3; }
out=results/stage43_vllm_awq_lora_dual_smoke
[[ ! -e "$out" ]] || { echo BLOCKED_RESULT_OVERWRITE;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
server_pid=''
cleanup(){
 local result=$?
 trap - EXIT INT TERM
 if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null;then kill -TERM "$server_pid";wait "$server_pid"||:;fi
 echo "STAGE43_ENGINE_CLEANUP_EXIT:$result"
 exit "$result"
}
trap cleanup EXIT INT TERM
bash scripts/run_stage43_vllm_lora_server.sh > "$out/server.log" 2>&1 & server_pid=$!
ready=0
for i in $(seq 1 160);do
 if curl -fsS --max-time 2 http://127.0.0.1:8033/health >/dev/null 2>&1;then ready=1;break;fi
 kill -0 "$server_pid" 2>/dev/null || { echo VLLM_DUAL_LORA_ENGINE_EXITED;tail -35 "$out/server.log";exit 4; }
 sleep 1
done
[[ "$ready" == 1 ]] || { echo DUAL_LORA_HEALTH_TIMEOUT;tail -40 "$out/server.log";exit 4; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader > "$out/gpu_during_server.csv"
timeout 500 python3 -m llm_opt_lab.serving.stage43_dual_adapter_smoke \
  --out "$out/real_dual_model_results.json" > "$out/client.log" 2>&1 || {
  echo DUAL_API_OR_ROUTING_FAILURE;tail -42 "$out/client.log";exit 5;
}
python3 - "$out" <<'PY'
import pathlib,hashlib,json,sys
root=pathlib.Path(sys.argv[1]);p=root/'real_dual_model_results.json';x=json.loads(p.read_text())
assert x['status']=='measured' and x['PSF_prompts']==24 and x['requests']==72
assert all(x['stats'][label]['completed']==24 for label in ('base','lora_sft','router'))
(root/'SHA256SUMS').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
print('STAGE43_REAL_AWQ_PLUS_LORA_DUAL_API_PASS',x['stats'])
PY
