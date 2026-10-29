#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD);approved=$(cat results/stage37_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_APPROVAL;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_MUTEX;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_PORT;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_OCCUPIED;exit 3; }
private=/home/you/projects/LLM-Training-Inference-Optimization-Lab/results/stage35_public_data
out=results/stage37_real_user_output384
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=.82 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
pid=''
cleanup(){ rc=$?;trap - EXIT INT TERM;if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid"||:;fi;echo STAGE37_CLEANUP:$rc;exit "$rc"; }
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
 local ok=0
 for i in $(seq 1 150);do
  if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ok=1;break;fi
  kill -0 "$pid" 2>/dev/null || { echo SERVER_FAILED;exit 4; }
  sleep 1
 done
 [[ "$ok" == 1 ]] || { echo SERVER_TIMEOUT;exit 4; }
}
for seed in 3511 3527 3559;do
 if [[ "$seed" == 3527 ]];then caches='on off';else caches='off on';fi
 for mix in cold0 hot50;do
  for cache in $caches;do
   id="seed${seed}_${mix}_rate4_${cache}"
   echo "STAGE37_START $id" | tee -a "$out/progress.log"
   start_server "$cache" "$id"
   timeout 350 python3 -m llm_opt_lab.serving.stage37_long_answer --seed "$seed" --mix "$mix" \
     --cache "$cache" --private "$private" --output "$out/${id}.json" \
     2>&1 | tee -a "$out/progress.log" || { echo STAGE37_CASE_INVALID:$id;exit 4; }
   stop_server
  done
 done
done
python3 - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('results/stage37_real_user_output384');files=sorted(root.glob('seed*.json'))
assert len(files)==12,len(files)
for f in files:
 x=json.loads(f.read_text());s=x['summary']
 assert x['status']=='measured' and x['config']['output_cap']==384
 assert s['offered']==s['completed']==96 and s['errors']==s['rejected']==0
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
print('STAGE37_LONG_ANSWER_384_PASS',len(files),len(files)*96,'real requests')
PY
