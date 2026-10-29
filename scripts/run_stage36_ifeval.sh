#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD);approved=$(cat results/stage36_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$approved" == "$sha" ]] || { echo BLOCKED_SOURCE_NOT_FROZEN;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_SERVER_ACTIVE;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY;exit 3; }
out=results/stage36_official_ifeval96
[[ ! -e "$out" ]] || { echo BLOCKED_RESULTS_EXIST;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src:$PWD/third_party/ifeval/google_research" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NLTK_DATA=/home/you/nltk_data
python=/home/you/venvs/llm-opt-qlora/bin/python
stage31=/home/you/projects/llmopt_stage31_frozen/results/stage31_full512_two_epochs
[[ -f "$stage31/selection.json" ]] || { echo BLOCKED_SELECTED_ADAPTER_MISSING;exit 3; }
timeout 3000 "$python" -m llm_opt_lab.training.stage36_ifeval --action generate \
 --input data --stage31 "$stage31" --out "$out/generations.json" > "$out/generation.log" 2>&1 || {
  echo STAGE36_GENERATE_FAILED;tail -35 "$out/generation.log";exit 4;
}
"$python" -m llm_opt_lab.training.stage36_ifeval --action score \
 --input data --stage31 "$stage31" --generated "$out/generations.json" \
 --out "$out/official_verifier_scores.json" > "$out/scoring.log" 2>&1 || {
  echo STAGE36_SCORE_FAILED;tail -35 "$out/scoring.log";exit 5;
}
python3 - "$out" <<'PY'
import pathlib,json,hashlib,sys
root=pathlib.Path(sys.argv[1]);a=json.loads((root/'generations.json').read_text());b=json.loads((root/'official_verifier_scores.json').read_text())
assert a['status']==b['status']=='measured'
assert len(a['cases'])==len(b['per_prompt'])==96
assert b['counts']['models']==3 and b['counts']['instruction_checks']>=96
assert b['generator_json_sha256']==hashlib.sha256((root/'generations.json').read_bytes()).hexdigest()
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in (root/'generations.json',root/'official_verifier_scores.json')))
print('STAGE36_OFFICIAL_GOOGLE_IFEVAL_96_VERIFIED',len(a['cases']),'x3',b['overall'])
PY
