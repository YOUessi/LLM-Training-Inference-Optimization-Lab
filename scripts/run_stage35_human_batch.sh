#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD);approved=$(cat results/stage35_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_SOURCE_SHA; exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE; exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_PORT;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY;exit 3; }
private=/home/you/projects/LLM-Training-Inference-Optimization-Lab/results/stage35_public_data
[[ -f "$private/stage35_selected_user_prompts.jsonl" ]] || { echo BLOCKED_PRIVATE_INPUT_MISSING;exit 3; }
out=results/stage35_frozen_real_user_batch
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING_RESULTS;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=.82 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
pid=''
cleanup(){ rc=$?;trap - EXIT INT TERM;if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid"||:;fi;echo STAGE35_CLEANUP:$rc;exit "$rc"; }
trap cleanup EXIT INT TERM
stop_server(){
 if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid"||:;fi;pid=''
 for i in $(seq 1 35);do if ! ss -ltn|grep -q ':8026 ';then break;fi;sleep 1;done
 ! ss -ltn|grep -q ':8026 ' || { echo STALE_PORT;exit 4; }
}
start_server(){
 local cache="$1" id="$2"
 export VLLM_PREFIX_CACHING="$cache"
 bash scripts/run_vllm_qwen3_awq.sh > "$out/server_${id}.log" 2>&1 & pid=$!
 local ready=0
 for i in $(seq 1 150);do
  if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
  kill -0 "$pid" 2>/dev/null || { echo SERVER_DIED;exit 4; }
  sleep 1
 done
 [[ "$ready" == 1 ]] || { echo SERVER_TIMEOUT;exit 4; }
}
for seed in 3511 3527 3559;do
 if [[ "$seed" == 3527 ]];then cache_order='on off';else cache_order='off on';fi
 for mix in cold0 hot50 hot100;do
  rates='8'
  if [[ "$mix" == hot50 ]];then rates='8 4';fi
  for rate in $rates;do
   for cache in $cache_order;do
    id="seed${seed}_${mix}_rate${rate}_${cache}"
    echo "STAGE35_RUN $id" | tee -a "$out/progress.log"
    start_server "$cache" "$id"
    timeout 300 python3 -m llm_opt_lab.serving.stage35_human_bench --mix "$mix" --seed "$seed" \
      --rate "$rate" --cache "$cache" --private "$private" --output "$out/${id}.json" \
      2>&1 | tee -a "$out/progress.log" || { echo FAIL_CLOSED:$id;exit 4; }
    stop_server
   done
  done
 done
done
python3 - <<'PY'
import json,hashlib,pathlib
root=pathlib.Path('results/stage35_frozen_real_user_batch');files=sorted(root.glob('seed*.json'))
if len(files)!=24:raise ValueError('expected 24 human-prompt cases got '+str(len(files)))
for p in files:
 x=json.loads(p.read_text());s=x['summary']
 if not (x['status']=='measured' and s['offered']==s['completed']==96 and s['errors']==s['rejected']==0):
  raise ValueError('invalid case '+str(p))
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
print('STAGE35_OASST_REAL_HUMAN_PASS',len(files),'cases',len(files)*96,'offers')
PY
