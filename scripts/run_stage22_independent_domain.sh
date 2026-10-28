#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY;exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo BLOCKED_PORT;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]' || { echo BLOCKED_GPU;exit 3; }
out=results/stage22_stdlib_qwen3_awq
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING;exit 3; }
mkdir -p "$out";git rev-parse HEAD > "$out/source_sha"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PREFIX_CACHING=off
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
pid=''
cleanup(){ rc=$?;trap - EXIT INT TERM;if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid" || :;fi;echo "STAGE22_CLEANUP $rc";exit "$rc"; }
trap cleanup EXIT INT TERM
bash scripts/run_vllm_qwen3_awq.sh > "$out/server.log" 2>&1 & pid=$!
ready=0
for i in $(seq 1 150);do
 if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
 kill -0 "$pid" 2>/dev/null || { echo SERVER_DIED;exit 4; }
 sleep 1
done
[[ "$ready" == 1 ]] || { echo HEALTH_TIMEOUT;exit 4; }
for seed in 9301 9309;do
 if [[ "$seed" == 9301 ]];then rates='8 16';policies='fifo_static400 sjf_static400 full_adaptive full_8b_calibrated';
 else rates='16 8';policies='full_8b_calibrated full_adaptive sjf_static400 fifo_static400';fi
 for rate in $rates;do
  for policy in $policies;do
   echo "STAGE22_GPU_CASE $seed $rate $policy" | tee -a "$out/progress.log"
   timeout 240 python3 -m llm_opt_lab.serving.stage22_stdlib_bench --model qwen3-8b-awq-local \
    --domain stdlib --seed "$seed" --rate "$rate" --policy "$policy" \
    --output "$out/stdlib_seed${seed}_rate${rate}_${policy}.json" | tee -a "$out/progress.log"
  done
 done
done
python3 - <<'PY'
import pathlib,json,hashlib
root=pathlib.Path('results/stage22_stdlib_qwen3_awq');files=sorted(root.glob('*.json'))
assert len(files)==16,len(files)
seen={}
for p in files:
 d=json.loads(p.read_text());c=d['config'];s=d['summary']
 assert d['status']=='measured' and s['offered']==24 and s['completed']+s['rejected']==24 and s['errors']==0,(p,d['status'])
 key=(c['seed'],c['rate_per_second']);ident=(c['arrival_schedule_offsets_s'],[r['request_id'] for r in d['requests']])
 if key in seen:assert seen[key]==ident
 else:seen[key]=ident
 print('VERIFIED',key,c['policy'],'SLO250',round(s['slo_all_offered']['250']['compliance_fraction'],3),'REJECTED',s['rejected'])
assert len(seen)==4
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
print('STAGE22_VALIDATED',len(files),'cases',len(files)*24,'offers')
PY
