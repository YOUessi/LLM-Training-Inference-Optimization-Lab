#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD);approved=$(cat results/stage44_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$approved" == "$sha" ]] || { echo BLOCKED_SOURCE_COMMIT; exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_UNFROZEN_SOURCE; exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_MUTEX;exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo BLOCKED_VLLM8026;exit 3; }
! ss -ltn | grep -q ':8033 ' || { echo BLOCKED_VLLM8033;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]' || { echo BLOCKED_GPU_USED;exit 3; }
private=/home/you/projects/LLM-Training-Inference-Optimization-Lab/results/stage44_ifbench_source
[[ -f "$private/stage44_ifbench_external60.jsonl" ]] || { echo BLOCKED_MISSING_VERIFIED_PRIVATE_SOURCE;exit 3; }
out=results/stage44_ifbench_external60
[[ ! -e "$out" ]] || { echo BLOCKED_PREEXISTING_OUTPUT;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src:$PWD/third_party/ifbench" NLTK_DATA=/home/you/nltk_data HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# IFBench untouched official evaluator auto prepends a private NLTK directory; keep it owner-only.
mkdir -p third_party/ifbench/ifbench/.nltk_data
chmod 700 third_party/ifbench/ifbench/.nltk_data
py=/home/you/venvs/llm-opt-qlora/bin/python
timeout 2200 "$py" -m llm_opt_lab.training.stage44_external_ifbench \
 --action generate --private "$private" --data data \
 --adapter evidence/stage31_full512/selected_adapter \
 --out "$out/local_private_full_generations.json" > "$out/gpu_generation.log" 2>&1 || {
 echo STAGE44_REAL_GENERATION_FAILED;tail -45 "$out/gpu_generation.log";exit 4;
}
timeout 600 "$py" -m llm_opt_lab.training.stage44_external_ifbench \
 --action score --private "$private" --data data \
 --generated "$out/local_private_full_generations.json" \
 --out "$out/official_scores_public.json" > "$out/offline_grading.log" 2>&1 || {
 echo STAGE44_OFFICIAL_VERIFIER_FAILED;tail -50 "$out/offline_grading.log";exit 5;
}
python3 - "$out" <<'PY'
import json,hashlib,pathlib,sys
folder=pathlib.Path(sys.argv[1]);gen=folder/'local_private_full_generations.json';sc=folder/'official_scores_public.json'
a=json.loads(gen.read_text());b=json.loads(sc.read_text())
assert a['status']==b['status']=='measured'
assert len(a['raw_outputs'])==len(b['public_anonymized_per_prompt_checks'])==60
assert b['original_generation_sha256']==hashlib.sha256(gen.read_bytes()).hexdigest()
assert set(b['aggregated'])=={'base','lora_sft','request_router'}
(folder/'PRIVATE_SHA256SUMS').write_text(hashlib.sha256(gen.read_bytes()).hexdigest()+'  local_private_full_generations.json\n')
(folder/'SHA256SUMS').write_text(hashlib.sha256(sc.read_bytes()).hexdigest()+'  official_scores_public.json\n')
print('STAGE44_REAL_EXTERNAL_IFBENCH_PASS',b['cross_benchmark_go'],b['aggregated'])
PY
