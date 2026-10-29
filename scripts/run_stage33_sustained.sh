#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD); approved=$(cat results/stage33_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_SOURCE_APPROVAL; exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE; exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK; exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_PORT; exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY; exit 3; }
out=results/stage33_sustained_real_8b
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING; exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PREFIX_CACHING=off VLLM_MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=.82 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
pid=''
cleanup(){ rc=$?;trap - EXIT INT TERM; if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid"||:;fi;echo STAGE33_CLEANUP:$rc; exit "$rc"; }
trap cleanup EXIT INT TERM
stop_server(){
 if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid"||:;fi;pid=''
 for i in $(seq 1 40);do if ! ss -ltn|grep -q ':8026 ';then break;fi;sleep 1;done
 ! ss -ltn|grep -q ':8026 ' || { echo STALE_PORT; exit 4; }
}
start_server(){
 local cache="$1" id="$2"
 export VLLM_PREFIX_CACHING="$cache"
 bash scripts/run_vllm_qwen3_awq.sh > "$out/server_${id}.log" 2>&1 & pid=$!
 local ready=0
 for i in $(seq 1 150);do
  if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
  kill -0 "$pid" 2>/dev/null || { echo SERVER_EXIT; exit 4; }
  sleep 1
 done
 [[ "$ready" == 1 ]] || { echo SERVER_TIMEOUT; exit 4; }
}
for seed in 3011 3029 3041;do
 if [[ "$seed" == 3029 ]];then cache_order='on off';else cache_order='off on';fi
 for mix in cold0 hot50;do
  for cache in $cache_order;do
   id="seed${seed}_${mix}_${cache}"
   echo "RUN_SUSTAINED $id" | tee -a "$out/progress.log"
   start_server "$cache" "$id"
   timeout 240 python3 -m llm_opt_lab.serving.stage33_sustained_bench --seed "$seed" --mix "$mix" \
     --cache "$cache" --output "$out/${id}.json" 2>&1 | tee -a "$out/progress.log" || { echo FAIL_CLOSED:$id; exit 4; }
   stop_server
  done
 done
done
python3 - <<'PY'
import pathlib,json,hashlib
r=pathlib.Path('results/stage33_sustained_real_8b');a=sorted(r.glob('seed*.json'))
assert len(a)==12,len(a)
for f in a:
 x=json.loads(f.read_text());s=x['summary']
 assert x['status']=='measured' and s['offered']==s['completed']==96 and s['errors']==s['rejected']==0
 assert x['config']['stage33_target_offered_window_s']==59.375
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in a))
print('STAGE33_60SECOND_PASS',len(a),'cases',len(a)*96,'real requests')
PY
