#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ -z "$(git status --porcelain)" ]] || { echo DIRTY;exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo PORT_OCCUPIED;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]' || { echo GPU_OCCUPIED;exit 3; }
out=results/stage15_qwen3_awq_pilot
[[ ! -e "$out" ]] || { echo OUTPUT_EXISTS;exit 3; }
mkdir -p "$out"
git rev-parse HEAD > "$out/source_sha"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PREFIX_CACHING=off
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
pid=''
cleanup(){ rc=$?;trap - EXIT INT TERM; if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null;then kill -TERM "$pid";wait "$pid" || :;fi;echo "STAGE15_CLEANUP $rc"; exit "$rc"; }
trap cleanup EXIT INT TERM
bash scripts/run_vllm_qwen3_awq.sh > "$out/server.log" 2>&1 & pid=$!
ready=0
for i in $(seq 1 150);do
 if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
 kill -0 "$pid" 2>/dev/null || { echo SERVER_DIED;tail -20 "$out/server.log";exit 4; }
 sleep 1
done
[[ "$ready" == 1 ]] || { echo TIMEOUT; exit 4; }
for policy in fifo_static400 sjf_static400 full_adaptive;do
 echo "AWQ_CASE $policy" | tee -a "$out/progress.log"
 timeout 250 python3 -m llm_opt_lab.serving.stage14_mixed_bench --model qwen3-8b-awq-local --domain k8s --seed 8109 --rate 8 --policy "$policy" --output "$out/rate8_${policy}.json" 2>&1 | tee -a "$out/progress.log"
done
python3 - <<'PY'
import pathlib,json,hashlib
r=pathlib.Path('results/stage15_qwen3_awq_pilot');fs=sorted(r.glob('*.json'));assert len(fs)==3
schedules=[]
for f in fs:
 d=json.loads(f.read_text());a=d['summary'];assert d['status']=='measured' and a['offered']==24 and a['errors']==0
 schedules.append((d['config']['arrival_schedule_offsets_s'],[x['request_id'] for x in d['requests']]))
 print('AWQ_VERIFIED',d['config']['policy'],a['completed'],a['rejected'],a['slo_all_offered']['250']['compliance_fraction'],a['useful_output_tokens_per_second'])
assert schedules[0]==schedules[1]==schedules[2]
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in fs))
PY
