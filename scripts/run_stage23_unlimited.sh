#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY;exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo BLOCKED_PORT;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU;exit 3; }
out=results/stage23_unlimited_qwen3_awq
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING;exit 3; }
mkdir -p "$out";git rev-parse HEAD > "$out/source_sha"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PREFIX_CACHING=off NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
pid=''
cleanup(){ x=$?;trap - EXIT INT TERM;if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid" || :;fi;echo "STAGE23_CLEANUP $x";exit "$x"; }
trap cleanup EXIT INT TERM
bash scripts/run_vllm_qwen3_awq.sh > "$out/server.log" 2>&1 & pid=$!
ready=0
for i in $(seq 1 150);do
 if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
 kill -0 "$pid" 2>/dev/null || exit 4
 sleep 1
done
[[ "$ready" == 1 ]] || exit 4
for seed in 9301 9309;do
 if [[ "$seed" == 9301 ]];then rates='8 16';else rates='16 8';fi
 for rate in $rates;do
  echo "STAGE23_GPU_CASE $seed $rate unlimited" | tee -a "$out/progress.log"
  timeout 240 python3 -m llm_opt_lab.serving.stage23_unlimited_bench --model qwen3-8b-awq-local --domain stdlib \
   --seed "$seed" --rate "$rate" --policy unlimited --output "$out/stdlib_seed${seed}_rate${rate}_unlimited.json" | tee -a "$out/progress.log"
 done
done
python3 - <<'PY'
import json,hashlib,pathlib
r=pathlib.Path('results/stage23_unlimited_qwen3_awq');fs=sorted(r.glob('*.json'));assert len(fs)==4
for f in fs:
 d=json.loads(f.read_text());s=d['summary'];assert d['status']=='measured' and s['offered']==24 and s['rejected']==0 and s['errors']==0
 print('VERIFIED',f.name,'250ms',s['slo_all_offered']['250']['compliance_fraction'])
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in fs))
PY
