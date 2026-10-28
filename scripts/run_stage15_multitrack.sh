#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -z "$(git status --porcelain)" ]] || { echo DIRTY;exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo PORT_OCCUPIED;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]' || { echo GPU_BUSY;exit 3; }
out=results/stage15_multitrack_20261029
[[ ! -e "$out" ]] || { echo ALREADY_EXECUTED;exit 3; }
mkdir -p "$out";git rev-parse HEAD > "$out/source_sha"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PREFIX_CACHING=off
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
pid=''
cleanup(){ rc=$?;trap - EXIT INT TERM;if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid"; wait "$pid" || :;fi; echo "MULTITRACK_CLEANUP $rc";exit "$rc";}
trap cleanup EXIT INT TERM
bash scripts/run_vllm_qwen3_awq.sh > "$out/server.log" 2>&1 & pid=$!
ready=0
for i in $(seq 1 150);do
 if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
 kill -0 "$pid" 2>/dev/null || { echo SERVER_DIED;exit 4; }
 sleep 1
done
[[ "$ready" == 1 ]] || { echo HEALTH_TIMEOUT;exit 4; }
for domain in k8s prose;do
 if [[ "$domain" == k8s ]];then seed=8101;else seed=8201;fi
 for rate in 8 16;do
  for policy in fifo_static400 sjf_static400 full_adaptive;do
   echo "CASE $domain $rate $policy" | tee -a "$out/progress.log"
   timeout 250 python3 -m llm_opt_lab.serving.stage14_mixed_bench --model qwen3-8b-awq-local --domain "$domain" --seed "$seed" --rate "$rate" --policy "$policy" --output "$out/${domain}_seed${seed}_rate${rate}_${policy}.json" 2>&1 | tee -a "$out/progress.log"
  done
 done
done
python3 - <<'PY'
import json,pathlib,hashlib
r=pathlib.Path('results/stage15_multitrack_20261029');fs=sorted(r.glob('*.json'));assert len(fs)==12,len(fs)
seen={}
for f in fs:
 d=json.loads(f.read_text());c=d['config'];s=d['summary'];assert d['status']=='measured' and s['offered']==24 and s['errors']==0
 key=(c['domain'],c['seed'],c['rate_per_second']);identity=(c['arrival_schedule_offsets_s'],[x['request_id'] for x in d['requests']])
 assert key not in seen or seen[key]==identity;seen[key]=identity
 print('VALID',*key,c['policy'],s['completed'],s['rejected'],s['slo_all_offered']['250']['compliance_fraction'])
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in fs))
print('STAGE15_MULTITRACK_PASS',len(fs),len(fs)*24)
PY
