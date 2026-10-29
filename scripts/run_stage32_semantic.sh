#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage32_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_SOURCE_SHA; exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE; exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_VLLM;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU;exit 3; }
out=results/stage32_semantic_greedy
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING_OUTPUT;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
stage31=/home/you/projects/llmopt_stage31_frozen/results/stage31_full512_two_epochs
[[ -f "$stage31/selected_final_test.json" && -f "$stage31/selection.json" ]] || { echo BLOCKED_STAGE31_NOT_COMPLETE;exit 4; }
# Real generation on NVIDIA GPU; NLI judgment on a separate CPU cross-encoder.
timeout 1800 /home/you/venvs/llm-opt-qlora/bin/python -m llm_opt_lab.training.stage32_semantic_quality \
 --action generate --stage31 "$stage31" --input data \
 --short-adapter evidence/stage26_sft_v2/selected_B_adapter \
 --out "$out/three_model_generations.json" > "$out/gpu_generation.log" 2>&1 || {
 tail -45 "$out/gpu_generation.log";exit 4;
}
# After Qwen3 generation is done, NLI runs on CPU in fresh process.
timeout 600 /home/you/venvs/llm-opt-qlora/bin/python -m llm_opt_lab.training.stage32_semantic_quality \
 --action nli --input data --generated "$out/three_model_generations.json" \
 --out "$out/semantic_nli.json" > "$out/nli_eval.log" 2>&1 || {
 tail -45 "$out/nli_eval.log";exit 5;
}
python3 - "$out" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);g=json.loads((r/'three_model_generations.json').read_text());n=json.loads((r/'semantic_nli.json').read_text())
assert g['status']=='measured' and n['status']=='measured'
assert g['config']['generated_count']==len(g['cases'])==64
assert n['counts']=={'items':64,'models':3,'nli_directional_pairs':384}
assert n['model_generations_sha256']==hashlib.sha256((r/'three_model_generations.json').read_bytes()).hexdigest()
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(r.glob('*.json'))))
print('STAGE32_SEMANTIC_PASS',64,'x',3,'outputs,',384,'NLI directional pairs')
PY
