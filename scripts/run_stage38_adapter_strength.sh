#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage38_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_UNAPPROVED_SOURCE;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_VLLM_SERVER;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_ALREADY_USED;exit 3; }
out=results/stage38_guarded_alpha_ifeval
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING_RESULT;exit 3; }
export PYTHONPATH="$PWD/src:$PWD/third_party/ifeval/google_research" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NLTK_DATA=/home/you/nltk_data
python=/home/you/venvs/llm-opt-qlora/bin/python
adapter=evidence/stage31_full512/selected_adapter
if [[ ! -f "$adapter/adapter_model.safetensors" ]];then echo BLOCKED_MISSING_SELECTED_ADAPTER;exit 3;fi
# Dev creates output itself, writes its decision BEFORE test. Capture logs out of output folder until created.
timeout 4800 "$python" -m llm_opt_lab.training.stage38_adapter_strength --action dev --data data \
 --adapter "$adapter" --out "$out" > /tmp/llmopt_stage38_dev_frozen.log 2>&1 || {
 echo STAGE38_DEV_FAILED;tail -48 /tmp/llmopt_stage38_dev_frozen.log;exit 4;
}
cp /tmp/llmopt_stage38_dev_frozen.log "$out/dev.log"
echo "$sha" > "$out/source_sha.txt"
"$python" - "$out/frozen_dev_selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='dev_approved_for_holdout_not_production' and x['new_dev_count']==43 and x['holdout_count']==101
assert x['holdout_not_loaded_or_used_for_selection']
print('STAGE38_DEV_PRE_TEST_DECISION_FROZEN',x['selected_alpha'],'feasible',x['guardrail_passed_with_adaptation'])
PY
# Only after selection file and SHA exist may the independent test action execute.
timeout 4600 "$python" -m llm_opt_lab.training.stage38_adapter_strength --action test --data data \
 --adapter "$adapter" --out "$out" > "$out/test.log" 2>&1 || {
 echo STAGE38_TEST_FAILED;tail -48 "$out/test.log";exit 5;
}
"$python" - "$out" <<'PY'
import json,hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]);s=json.loads((root/'frozen_dev_selection.json').read_text());x=json.loads((root/'final_untouched101_test.json').read_text())
assert s['selected_alpha']==x['selected_alpha'] and s['holdout_not_loaded_or_used_for_selection']
assert x['status']=='measured' and x['new_holdout_count']==101 and x['dev43_disjoint']
assert x['dev_decision_sha256']==hashlib.sha256((root/'frozen_dev_selection.json').read_bytes()).hexdigest()
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in sorted(root.glob('*.json'))))
print('STAGE38_FINAL_UNTOUCHED101_VERIFIED',x['selected_and_comparators'])
PY
