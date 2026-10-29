#!/usr/bin/env bash
# Freeze source SHA and finite fixed cases; run on an isolated detached worktree.
set -euo pipefail
cd "$(dirname "$0")/.."
source_sha="$(git rev-parse HEAD)"
approved_sha="$(cat results/stage25_approved_source_sha.txt 2>/dev/null || :)"
[[ -n "$approved_sha" && "$source_sha" == "$approved_sha" ]] || { echo BLOCKED_SOURCE_NOT_APPROVED;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9> /home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCKED;exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo BLOCKED_PORT_8026;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_COMPUTE;exit 3; }
out=results/stage25_frozen_batch
[[ ! -e "$out" ]] || { echo BLOCKED_NO_OVERWRITE;exit 3; }
mkdir -p "$out"
printf '%s\n' "$source_sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
server_pid=''
cleanup(){ code=$?;trap - EXIT INT TERM;if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null;then kill -TERM "$server_pid"; wait "$server_pid" || :; fi; echo "STAGE25_CLEANUP $code";exit "$code"; }
trap cleanup EXIT INT TERM
stop_server(){
 if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null;then kill -TERM "$server_pid";wait "$server_pid" || :;fi
 server_pid=''
 for i in $(seq 1 40);do
  if ! ss -ltn | grep -q ':8026 ';then break;fi
  sleep 1
 done
 ! ss -ltn | grep -q ':8026 ' || { echo BLOCKED_STALE_PORT;exit 4; }
}
launch_server(){
 local cap="$1" cache="$2" util="$3"
 export VLLM_MAX_NUM_SEQS="$cap" VLLM_PREFIX_CACHING="$cache" GPU_MEMORY_UTILIZATION="$util"
 echo "ENGINE_START cap=$cap prefix=$cache gpu_util=$util" | tee -a "$out/progress.log"
 bash scripts/run_vllm_qwen3_awq.sh > "$out/server_cap${cap}_cache${cache}_util${util}.log" 2>&1 & server_pid=$!
 local ready=0
 for i in $(seq 1 160);do
  if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
  kill -0 "$server_pid" 2>/dev/null || { echo BLOCKED_ENGINE_DIED;exit 4; }
  sleep 1
 done
 [[ "$ready" == 1 ]] || { echo BLOCKED_ENGINE_HEALTH_TIMEOUT;exit 4; }
}
run_case(){
 local cohort="$1" seed="$2" rate="$3" capout="$4" cap="$VLLM_MAX_NUM_SEQS" cache="$VLLM_PREFIX_CACHING" util="$GPU_MEMORY_UTILIZATION"
 local filepath="$out/${cohort}_s${seed}_r${rate}_cap${cap}_cache${cache}_util${util}_out${capout}.json"
 [[ ! -e "$filepath" ]] || { echo BLOCKED_EXISTING_CASE;exit 3; }
 echo "CASE $cohort seed=$seed rate=$rate cap=$cap cache=$cache util=$util out=$capout" | tee -a "$out/progress.log"
 timeout 300 python3 -m llm_opt_lab.serving.stage25_capacity_bench --domain "$cohort" --seed "$seed" --rate "$rate" \
  --output-cap "$capout" --output "$filepath" 2>&1 | tee -a "$out/progress.log"
 python3 - "$filepath" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));s=p['summary']
assert p['status']=='measured' and s['offered']==24 and s['errors']==0 and s['completed']==24 and s['rejected']==0
assert p['server_queue_monitor']['available'] and p['kv_cache_monitor']['available']
print('VALID_CASE',p['config']['seed'],p['config']['rate_per_second'],p['config']['stage25_workload_domain'],p['config']['stage25_output_cap'])
PY
}
for cap in 8 16;do
 for cache in off on;do
  launch_server "$cap" "$cache" .82
  for seed in 9401 9409;do
   for rate in 8 16;do run_case shared "$seed" "$rate" 128;done
  done
  if [[ "$cap" == 16 ]];then
   run_case distinct 9401 16 128
   if [[ "$cache" == off ]];then
    run_case shared 9401 16 256
    run_case shared 9409 16 256
   fi
  fi
  stop_server
 done
done
launch_server 16 off .70
for seed in 9401 9409;do run_case shared "$seed" 16 128;done
stop_server
python3 - <<'PY'
import pathlib,json,hashlib
r=pathlib.Path('results/stage25_frozen_batch');paths=sorted(r.glob('*.json'))
if len(paths)!=22:raise ValueError('not all 22 real cases: '+str(len(paths)))
for p in paths:
 d=json.loads(p.read_text());s=d['summary']
 if d['status']!='measured' or s['offered']!=24 or s['errors'] or s['completed']!=24:raise ValueError('bad result '+str(p))
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in paths))
print('STAGE25_FROZEN_BATCH_PASS',len(paths),'cases',len(paths)*24,'offers')
PY
