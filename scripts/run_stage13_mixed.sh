#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

domain="$1"
seed="$2"
case "$domain:$seed" in
 k8s:6101|k8s:6109|prose:7201|prose:7209) ;;
 *) echo "BLOCKED: requires frozen domain/seed pair" >&2; exit 3;;
esac

out="results/stage13_"$domain"_seed"$seed
if [[ -e "$out" ]];then
 echo "BLOCKED: existing result path $out" >&2;exit 3
fi
if [[ -n "$(git status --porcelain)" ]];then
 echo "BLOCKED: tracked worktree is dirty" >&2;exit 3
fi
if ss -ltn | grep -q ':8026 ';then
 echo "BLOCKED: localhost model port already in use" >&2;exit 3
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null |grep -Eq '[0-9]';then
 echo "BLOCKED: another GPU compute process running" >&2;exit 3
fi

mkdir -p "$out"
umask 077
export PYTHONPATH="$PWD/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_PREFIX_CACHING=off
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
unset VLLM_NUM_GPU_BLOCKS_OVERRIDE
server_pid=""
cleanup() {
 status=$?
 trap - EXIT INT TERM
 if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null;then
  kill -TERM "$server_pid" 2>/dev/null || :
  wait "$server_pid" 2>/dev/null || :
 fi
 echo "STAGE13_SERVER_CLEANUP status=$status"
 exit "$status"
}
trap cleanup EXIT INT TERM

echo "SOURCE_COMMIT $(git rev-parse HEAD)"
echo "STAGE13 domain=$domain seed=$seed"
nvidia-smi --query-gpu=name,memory.used,driver_version --format=csv,noheader > "$out/gpu_before.txt"
bash scripts/run_vllm_local.sh > "$out/server.log" 2>&1 &
server_pid=$!
ready=0
for i in $(seq 1 120);do
 if curl -fsS --max-time 3 http://127.0.0.1:8026/health >/dev/null 2>&1;then
  ready=1;break
 fi
 if ! kill -0 "$server_pid" 2>/dev/null;then
  echo "BLOCKED: vLLM server exited unexpectedly" >&2
  tail -24 "$out/server.log" >&2
  exit 4
 fi
 sleep 1
done
if [[ "$ready" -ne 1 ]];then
 echo "BLOCKED: serving readiness timeout" >&2;exit 4
fi

curl -fsS --max-time 5 http://127.0.0.1:8026/v1/models > "$out/models.json"
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_initial.prom"
if [[ "$seed" == 6101 || "$seed" == 7201 ]];then
 rate_order="8 16"
 policy_order="unlimited fixed150 fixed400 adaptive"
else
 rate_order="16 8"
 policy_order="adaptive fixed400 fixed150 unlimited"
fi

for rate in $rate_order;do
 for policy in $policy_order;do
  filename="rate"$rate"_"$policy"_seed"$seed".json"
  echo "STAGE13_GPU_CASE domain=$domain seed=$seed offered_rate=$rate policy=$policy"
  timeout 170 python3 -m llm_opt_lab.serving.stage13_mixed_bench \
    --model qwen2.5-1.5b-instruct-local --domain "$domain" \
    --seed "$seed" --rate "$rate" --policy "$policy" \
    --output "$out/$filename"
 done
done
curl -fsS --max-time 5 http://127.0.0.1:8026/metrics > "$out/metrics_final.prom"
python3 - "$domain" "$seed" "$out" <<'PY'
import json,sys
from pathlib import Path
domain,seed,out=sys.argv[1],int(sys.argv[2]),Path(sys.argv[3])
count=complete=reject=0
for rate in (8,16):
 for policy in ("unlimited","fixed150","fixed400","adaptive"):
  p=out/f"rate{rate}_{policy}_seed{seed}.json"
  d=json.loads(p.read_text())
  c=d["config"];summary=d["summary"]
  assert d["status"]=="measured" and d["environment"]["git_commit"]
  assert c["domain"]==domain and c["seed"]==seed
  assert (c["rate_per_second"],c["policy"])==(rate,policy)
  assert summary["offered"]==24
  assert summary["completed"]+summary["rejected"]==24 and summary["errors"]==0
  assert summary["accepted"]==summary["completed"]
  assert len(d["requests"])==24 and d["server_queue_monitor"]["available"]
  assert d["gate_final_state"]["active"]==d["gate_final_state"]["waiters"]==0
  assert sum(x["output_mode"]=="label" for x in d["requests"])==12
  assert sum(x["output_mode"]=="detailed" for x in d["requests"])==12
  count+=summary["offered"]
  complete+=summary["completed"]
  reject+=summary["rejected"]
  print("CASE",domain,seed,rate,policy,
        "complete",summary["completed"],"reject",summary["rejected"],
        "SLO250",round(summary["slo_all_offered"]["250"]["compliance_fraction"],3),
        "useful_TPS",round(summary["useful_output_tokens_per_second"],1),
        "clientwait_P95",round(summary["p95_admission_queue_wait_ms"],1))
assert count==192 and count==complete+reject
print("STAGE13_SEED_COMPLETE",domain,seed,"offered",count,
      "completed",complete,"explicit_rejected",reject)
PY
echo "STAGE13_REAL_GPU_DONE $domain $seed"
