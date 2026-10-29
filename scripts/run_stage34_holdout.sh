#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD);approved=$(cat results/stage34_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_UNAPPROVED_SOURCE;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_SERVER_LISTENER;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY;exit 3; }
out=results/stage34_new128_dolly_holdout
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
stage31=/home/you/projects/llmopt_stage31_frozen/results/stage31_full512_two_epochs
[[ -f "$stage31/selected_final_test.json" && -f "$stage31/selection.json" ]] || { echo BLOCKED_NOT_SELECTED;exit 4; }
timeout 420 /home/you/venvs/llm-opt-qlora/bin/python -m llm_opt_lab.training.stage34_fresh_holdout_eval \
 --input data --stage31 "$stage31" --stage26-adapter evidence/stage26_sft_v2/selected_B_adapter \
 --out "$out/fresh128_three_model_CE.json" > "$out/eval.log" 2>&1 || { tail -35 "$out/eval.log";exit 4; }
python3 - "$out" <<'PY'
import json,hashlib,sys,pathlib
r=pathlib.Path(sys.argv[1]);x=json.loads((r/'fresh128_three_model_CE.json').read_text())
assert x['status']=='measured' and x['counts']=={'examples':128,'categories':8,'per_category':16,'models':3}
assert x['source_ids_excluded_from_all_previous_704']
(r/'SHA256SUMS').write_text(hashlib.sha256((r/'fresh128_three_model_CE.json').read_bytes()).hexdigest()+'  fresh128_three_model_CE.json\n')
print('STAGE34_FRESH_UNTOUCHED_128_PASS',x['overall'])
PY
