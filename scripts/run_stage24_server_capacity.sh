#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY;exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo BLOCKED_PORT;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU;exit 3; }
out=results/stage24_server_capacity_qwen3_awq
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING;exit 3; }
mkdir -p "$out";git rev-parse HEAD > "$out/source_sha"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PREFIX_CACHING=off NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
server=''
stop_server(){ if [[ -n "$server" ]] && kill -0 "$server" 2>/dev/null;then kill -TERM "$server";wait "$server" || :;fi;server=''; }
cleanup(){ rc=$?;trap - EXIT INT TERM;stop_server;echo "STAGE24_CLEANUP $rc";exit "$rc"; }
trap cleanup EXIT INT TERM
for seed in 9301 9309;do
 if [[ "$seed" == 9301 ]];then caps='8 16';rates='8 16';else caps='16 8';rates='16 8';fi
 for cap in $caps;do
  export VLLM_MAX_NUM_SEQS="$cap"
  echo "STAGE24_START_SERVER seed=$seed capacity=$cap" | tee -a "$out/progress.log"
  bash scripts/run_vllm_qwen3_awq.sh > "$out/server_${seed}_${cap}.log" 2>&1 & server=$!
  ready=0
  for i in $(seq 1 150);do
   if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
   kill -0 "$server" 2>/dev/null || { echo SERVER_DIED;exit 4; }
   sleep 1
  done
  [[ "$ready" == 1 ]] || { echo SERVER_HEALTH_TIMEOUT;exit 4; }
  for rate in $rates;do
   echo "STAGE24_GPU_CASE seed=$seed rate=$rate maxseq=$cap" | tee -a "$out/progress.log"
   timeout 260 python3 -m llm_opt_lab.serving.stage23_unlimited_bench --model qwen3-8b-awq-local \
    --domain stdlib --seed "$seed" --rate "$rate" --policy unlimited \
    --output "$out/stdlib_seed${seed}_rate${rate}_maxseq${cap}_unlimited.json" | tee -a "$out/progress.log"
  done
  stop_server
  for i in $(seq 1 20);do
    if ! ss -ltn | grep -q ':8026 ';then break;fi
    sleep 1
  done
  ! ss -ltn | grep -q ':8026 ' || { echo PORT_NOT_RELEASED;exit 4; }
 done
done
python3 - <<'PY'
import json,hashlib,pathlib
r=pathlib.Path('results/stage24_server_capacity_qwen3_awq');fs=sorted(r.glob('*.json'));assert len(fs)==8,len(fs)
cells={}
for f in fs:
 d=json.loads(f.read_text());c=d['config'];s=d['summary']
 assert d['status']=='measured' and s['offered']==24 and s['completed']==24 and s['rejected']==s['errors']==0
 key=(c['seed'],c['rate_per_second']);cap=c['server_max_num_seqs'];assert cap in (8,16)
 ident=(c['arrival_schedule_offsets_s'],[x['request_id'] for x in d['requests']])
 if key in cells:assert cells[key][1]==ident and cells[key][0]!=cap
 cells[key]=(cap,ident)
 print('VERIFIED',key,'CAP',cap,'SLO250',round(s['slo_all_offered']['250']['compliance_fraction'],3),'TPS',round(s['useful_output_tokens_per_second'],1))
assert len(cells)==4
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in fs))
print('STAGE24_REAL_GPU_PASS',len(fs),'cases',len(fs)*24,'offers')
PY
