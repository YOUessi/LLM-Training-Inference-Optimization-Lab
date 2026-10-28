#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY; exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo BLOCKED_PORT; exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]' || { echo BLOCKED_GPU; exit 3; }
out=results/stage14_batch_20261029
[[ ! -e "$out" ]] || { echo BLOCKED_OUTPUT_EXISTS; exit 3; }
mkdir -p "$out"
git rev-parse HEAD > "$out/source_sha"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PREFIX_CACHING=off
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
pid=''
cleanup(){ rc=$?; trap - EXIT INT TERM; if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid"; wait "$pid" || :;fi; echo "BATCH_CLEANUP $rc"; exit "$rc"; }
trap cleanup EXIT INT TERM
bash scripts/run_vllm_local.sh > "$out/server.log" 2>&1 & pid=$!
ready=0
for i in $(seq 1 120);do
 if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
 kill -0 "$pid" 2>/dev/null || { echo SERVER_EXIT; exit 4; }
 sleep 1
done
[[ "$ready" == 1 ]] || { echo NOT_READY; exit 4; }
# 2 domains x 2 arrival rates x 8 policies = 32 independently saved cases.
for domain in k8s prose;do
 if [[ "$domain" == k8s ]];then seed=8109;else seed=8201;fi
 for rate in 16 8;do
  for policy in fifo_static400 fifo_static400_ewma sjf_static400_noewma sjf_static400 sjf_aging_static400 fifo_dynamic sjf_dynamic full_adaptive;do
   target="$out/${domain}_seed${seed}_rate${rate}_${policy}.json"
   echo "CASE $domain $seed $rate $policy" | tee -a "$out/progress.log"
   timeout 210 python3 -m llm_opt_lab.serving.stage14_mixed_bench --model qwen2.5-1.5b-instruct-local --domain "$domain" --seed "$seed" --rate "$rate" --policy "$policy" --output "$target" 2>&1 | tee -a "$out/progress.log"
  done
 done
done
python3 - <<'PY'
import json,pathlib,hashlib
r=pathlib.Path('results/stage14_batch_20261029');paths=sorted(r.glob('*.json'))
assert len(paths)==32,len(paths)
bycell={}
for p in paths:
 d=json.loads(p.read_text()); c=d['config'];s=d['summary'];key=(c['domain'],c['seed'],c['rate_per_second'])
 assert d['status']=='measured' and s['offered']==24 and s['errors']==0 and s['completed']+s['rejected']==24
 ident=([x['request_id'] for x in d['requests']],c['arrival_schedule_offsets_s'])
 if key in bycell:assert bycell[key]==ident
 else:bycell[key]=ident
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in paths))
print('BATCH_VALIDATED',len(paths),'cases',len(paths)*24,'offers',len(bycell),'matched conditions')
PY
