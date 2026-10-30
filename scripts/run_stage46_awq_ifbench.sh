#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage46_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_UNAPPROVED_SOURCE_SHA;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_SHARED_MUTEX;exit 3; }
! ss -ltn | grep -q ':8033 ' || { echo BLOCKED_ENGINE_PORT;exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo BLOCKED_SECONDARY_VLLM_PORT;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]' || { echo BLOCKED_GPU_ACTIVE;exit 3; }
private=/home/you/projects/LLM-Training-Inference-Optimization-Lab/results/stage44_ifbench_source
[[ -f "$private/stage44_ifbench_external60.jsonl" ]] || { echo BLOCKED_IFBENCH_ORIGINAL_SOURCE_MISSING;exit 3; }
out=results/stage46_awq_real_ifbench60
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING_OUTPUT;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src:$PWD/third_party/ifbench" NLTK_DATA=/home/you/nltk_data HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
mkdir -p third_party/ifbench/ifbench/.nltk_data
chmod 700 third_party/ifbench/ifbench/.nltk_data
server_pid=''
cleanup(){ rc=$?;trap - EXIT INT TERM;if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null;then kill -TERM "$server_pid";wait "$server_pid"||:;fi;echo STAGE46_VLLM_CLEANUP_EXIT:$rc;exit "$rc"; }
trap cleanup EXIT INT TERM
bash scripts/run_stage43_vllm_lora_server.sh > "$out/server.log" 2>&1 & server_pid=$!
ready=0
for i in $(seq 1 165);do
 if curl -fsS --max-time 2 http://127.0.0.1:8033/health >/dev/null 2>&1;then ready=1;break;fi
 kill -0 "$server_pid" 2>/dev/null || { echo STAGE46_DUAL_MODEL_SERVER_FAILED;tail -25 "$out/server.log";exit 4; }
 sleep 1
done
[[ "$ready" == 1 ]] || { echo STAGE46_ENGINE_HEALTH_TIMEOUT;exit 4; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$out/allocated_gpu.csv"
py=/home/you/venvs/llm-opt-qlora/bin/python
timeout 1800 "$py" -m llm_opt_lab.serving.stage46_vllm_ifeval_quality --action generate \
 --private "$private" --data data --out "$out/private_generation.json" > "$out/generate.log" 2>&1 || {
 echo STAGE46_AWQ_GENERATION_FAILED;tail -35 "$out/generate.log";exit 4;
}
timeout 600 "$py" -m llm_opt_lab.serving.stage46_vllm_ifeval_quality --action score \
 --private "$private" --data data --generated "$out/private_generation.json" \
 --out "$out/official_scores.json" > "$out/score.log" 2>&1 || {
 echo STAGE46_IFBENCH_OFFICIAL_SCORING_FAILED;tail -45 "$out/score.log";exit 5;
}
python3 - "$out" <<'PY'
import pathlib,json,hashlib,sys
root=pathlib.Path(sys.argv[1]);priv=root/'private_generation.json';pub=root/'official_scores.json'
a=json.loads(priv.read_text());b=json.loads(pub.read_text())
assert a['status']==b['status']=='measured'
assert len(a['raw_outputs'])==len(b['public_anonymized_per_prompt_checks'])==60
assert b['original_generation_sha256']==hashlib.sha256(priv.read_bytes()).hexdigest()
assert b['kind']=='stage46_official_IFBench60_AWQ_vLLM_base_plus_LoRA'
(root/'SHA256SUMS').write_text(hashlib.sha256(pub.read_bytes()).hexdigest()+'  official_scores.json\n')
(root/'PRIVATE_SHA256SUMS').write_text(hashlib.sha256(priv.read_bytes()).hexdigest()+'  private_generation.json\n')
print('STAGE46_VERIFIED_120_REAL_VLLM_AWQ_IFBENCH_API_OUTPUTS',b['aggregated'],b['cross_benchmark_go'])
PY
