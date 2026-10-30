#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD);approved=$(cat results/stage45_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_UNAPPROVED_SHA;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_GIT;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_VLLM8026;exit 3; }
! ss -ltn|grep -q ':8033 ' || { echo BLOCKED_VLLM8033;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_EXISTING_GPU;exit 3; }
private=/home/you/projects/LLM-Training-Inference-Optimization-Lab/results/stage35_public_data
[[ -f "$private/stage35_selected_user_prompts.jsonl" ]] || { echo BLOCKED_MISSING_PRIVACY_PROTECTED_SOURCE;exit 3; }
out=results/stage45_real_dual_concurrent
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING_RESULT;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
pid=''
cleanup(){ rc=$?;trap - EXIT INT TERM; if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid"||:;fi;echo STAGE45_CLEANUP_EXIT:$rc;exit "$rc"; }
trap cleanup EXIT INT TERM
stop_server(){
 if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid"||:;fi
 pid=''
 for i in $(seq 1 35);do if ! ss -ltn|grep -q ':8033 ';then break;fi;sleep 1;done
 ! ss -ltn|grep -q ':8033 ' || { echo STALE_DUAL_SERVER;exit 4; }
}
start_server(){
 local name="$1";local ok=0
 bash scripts/run_stage43_vllm_lora_server.sh > "$out/server_${name}.log" 2>&1 & pid=$!
 for i in $(seq 1 165);do
  if curl -fsS --max-time 2 http://127.0.0.1:8033/health >/dev/null 2>&1;then ok=1;break;fi
  kill -0 "$pid" 2>/dev/null || { echo MODEL_SERVER_DIED;tail -32 "$out/server_${name}.log";exit 4; }
  sleep 1
 done
 [[ "$ok" == 1 ]] || { echo DUAL_MODEL_SERVER_TIMEOUT;exit 4; }
}
for seed in 4511 4527;do
 for rate in 4 8;do
  if [[ "$seed" == 4527 ]];then order='request_router lora_sft base';else order='base lora_sft request_router';fi
  for policy in $order;do
   id="seed${seed}_rate${rate}_${policy}"
   echo "STAGE45_DUAL_VLLM_ARM_START $id" | tee -a "$out/progress.log"
   start_server "$id"
   timeout 420 python3 -m llm_opt_lab.serving.stage45_router_load --seed "$seed" --rate "$rate" \
     --policy "$policy" --private "$private" --input data --output "$out/${id}.json" \
     2>&1 | tee -a "$out/progress.log" || { echo STAGE45_FAIL_CLOSED:$id;exit 5; }
   stop_server
  done
 done
done
python3 - <<'PY'
from pathlib import Path
import json,hashlib
r=Path('results/stage45_real_dual_concurrent');files=sorted(r.glob('seed*.json'))
if len(files)!=12:raise ValueError(f'expected 12 real arms, got {len(files)}')
for f in files:
 x=json.loads(f.read_text());s=x['summary']
 if not(x['status']=='measured' and s['offered']==s['completed']==72 and s['errors']==s['rejected']==0):
  raise ValueError('invalid real GPU request arm '+str(f))
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in files))
print('STAGE45_REAL_AWQ_LORA_ROUTER_CONCURRENT_PASS',len(files),len(files)*72,'actual requests')
PY
