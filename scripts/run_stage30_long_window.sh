#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage30_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_SOURCE_SHA; exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY; exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK; exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_PORT; exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_OCCUPIED; exit 3; }
out=results/stage30_frozen_mixed_8b
[[ ! -e "$out" ]] || { echo BLOCKED_PREEXISTING_OUTPUT; exit 3; }
mkdir -p "$out"; echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 VLLM_MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=.82
pid=''
cleanup(){rc=$?;trap - EXIT INT TERM;if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid" || :; fi; echo STAGE30_CLEANUP:$rc; exit "$rc"; }
trap cleanup EXIT INT TERM
stop_server(){
 if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid"||:; fi
 pid=''
 for i in $(seq 1 35);do if ! ss -ltn|grep -q ':8026 ';then break; fi;sleep 1;done
 ! ss -ltn|grep -q ':8026 ' || { echo STALE_PORT; exit 4; }
}
start_server(){
 local cache="$1" id="$2"
 export VLLM_PREFIX_CACHING="$cache"
 echo "SERVER_START cache=$cache arm=$id" | tee -a "$out/progress.log"
 bash scripts/run_vllm_qwen3_awq.sh > "$out/server_${id}.log" 2>&1 & pid=$!
 ready=0
 for i in $(seq 1 150);do
  if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break; fi
  kill -0 "$pid" 2>/dev/null || { echo SERVER_EXIT; tail -20 "$out/server_${id}.log"; exit 4; }
  sleep 1
 done
 [[ "$ready" == 1 ]] || { echo SERVER_HEALTH_TIMEOUT; exit 4; }
}
for seed in 3011 3029 3041;do
 if [[ "$seed" == 3029 ]];then caches='on off';else caches='off on'; fi
 for mix in cold0 hot50 hot100;do
  for cache in $caches;do
   id="seed${seed}_${mix}_${cache}"
   start_server "$cache" "$id"
   echo "RUN_CASE $id" | tee -a "$out/progress.log"
   timeout 300 python3 -m llm_opt_lab.serving.stage30_mixed_bench --seed "$seed" --mix "$mix" --rate 8 \
     --prefix-cache "$cache" --model qwen3-8b-awq-local --output "$out/${id}.json" 2>&1 | tee -a "$out/progress.log" || {
     echo "STAGE30_FAIL_CLOSED $id"; exit 4;
   }
   stop_server
  done
 done
done
python3 - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('results/stage30_frozen_mixed_8b'); files=sorted(r.glob('*.json'))
if len(files)!=18:raise ValueError('incomplete cases '+str(len(files)))
for f in files:
 x=json.loads(f.read_text());s=x['summary']
 if not(x['status']=='measured' and s['offered']==s['completed']==96 and s['rejected']==s['errors']==0):raise ValueError('invalid case '+str(f))
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in files))
print('STAGE30_REAL_BATCH_PASS',len(files),'cases',96*len(files),'offered')
PY
