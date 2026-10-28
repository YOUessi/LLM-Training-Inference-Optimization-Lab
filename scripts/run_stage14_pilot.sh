#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain)" ]]; then echo 'BLOCKED dirty tracked source' >&2;exit 3;fi
if ss -ltn | grep -q ':8026 ';then echo 'BLOCKED service already present' >&2;exit 3;fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]';then echo 'BLOCKED GPU occupied' >&2;exit 3;fi
if [[ -d results/stage14_pilot_k8s_seed8101 ]];then echo 'BLOCKED output already exists' >&2;exit 3;fi
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_PREFIX_CACHING=off
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
out=results/stage14_pilot_k8s_seed8101
mkdir -p "$out"
printf '%s\n' "$(git rev-parse HEAD)" > "$out/source_git_sha.txt"
server_pid=''
cleanup(){ rc=$?; trap - EXIT INT TERM; if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null;then kill -TERM "$server_pid";wait "$server_pid" || :;fi; echo "STAGE14_PILOT_CLEANUP $rc";exit "$rc";}
trap cleanup EXIT INT TERM
bash scripts/run_vllm_local.sh > "$out/server.log" 2>&1 & server_pid=$!
ready=0
for i in $(seq 1 120);do
 if curl -fsS --max-time 2 http://127.0.0.1:8026/health >/dev/null 2>&1;then ready=1;break;fi
 if ! kill -0 "$server_pid" 2>/dev/null;then echo 'BLOCKED server died' >&2;exit 4;fi
 sleep 1
done
[[ "$ready" == 1 ]] || { echo 'BLOCKED health timeout' >&2;exit 4; }
for policy in fifo_static400 full_adaptive;do
  echo "STAGE14_PILOT_REAL_REQUEST $policy" | tee -a "$out/progress.log"
  timeout 210 python3 -m llm_opt_lab.serving.stage14_mixed_bench --model qwen2.5-1.5b-instruct-local --domain k8s --seed 8101 --rate 16 --policy "$policy" --output "$out/rate16_${policy}.json" 2>&1 | tee -a "$out/progress.log"
done
python3 - <<'PY'
import json,hashlib,pathlib
root=pathlib.Path('results/stage14_pilot_k8s_seed8101')
for policy in ('fifo_static400','full_adaptive'):
 p=root/f'rate16_{policy}.json';d=json.loads(p.read_text());s=d['summary']
 assert d['status']=='measured' and s['offered']==24 and s['errors']==0 and s['completed']+s['rejected']==24
 print('VERIFIED',policy,'completed',s['completed'],'rejected',s['rejected'],'slo250',s['slo_all_offered']['250']['compliance_fraction'],'tps',s['useful_output_tokens_per_second'])
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(root.glob('*.json'))))
PY
