#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage40_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_SOURCE_SHA;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_SERVING;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY;exit 3; }
out=results/stage40_real_constraint_rehearsal
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING_RESULTS;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha"
export PYTHONPATH="$PWD/src:$PWD/third_party/ifeval/google_research" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NLTK_DATA=/home/you/nltk_data
py=/home/you/venvs/llm-opt-qlora/bin/python
for seed in 4011 4027;do
 for part in '00:0' '25:0.25' '50:0.5';do
  name="mix${part%%:*}_seed${seed}"
  mix="${part#*:}"
  echo "STAGE40_GPU_TRAIN $name" | tee -a "$out/progress.log"
  timeout 1800 "$py" -m llm_opt_lab.training.stage40_constraint_rehearsal \
   --data data --seed "$seed" --mix "$mix" \
   --adapter "$out/${name}_adapter" --out "$out/${name}.json" > "$out/${name}.log" 2>&1 || {
     echo "STAGE40_FAILED_TRAIN $name";tail -45 "$out/${name}.log";exit 4;
   }
  "$py" - "$out/${name}.json" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]))
assert x['status']=='measured' and x['config']['source_exposures']==1024 and x['config']['optimizer_updates']==512
assert len(x['per_epoch'])==2 and all(e['unique_source_records']==512 for e in x['per_epoch'])
assert x['best_epoch_by_dolly_validation_only'] in (1,2)
assert math.isfinite(x['best_validation_ce'])
print('STAGE40_TRAIN_VALID',x['config']['mix'],x['config']['seed'],x['best_validation_ce'])
PY
 done
done
# Zero IFEval test prompts read until after dev decision is durably finalized.
echo STAGE40_IFEVAL_DEV_GUARDRAIL_START | tee -a "$out/progress.log"
timeout 4500 "$py" -m llm_opt_lab.training.stage40_policy_eval --action dev --data data --root "$out" \
 > "$out/dev_eval.log" 2>&1 || { echo BLOCKED_DEV_EVALUATION;tail -45 "$out/dev_eval.log";exit 5; }
"$py" - "$out/dev_selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['status']=='dev_frozen_no_final_test_access' and x['final_test_unaccessed']
assert x['untouched60_sha256'] and len(x['training_candidates'])==6
print('STAGE40_DEV_SELECTION_FROZEN',x['selected'],'feasible',x['feasible_adapted'])
PY
echo STAGE40_UNTOUCHED60_TEST_START | tee -a "$out/progress.log"
timeout 4000 "$py" -m llm_opt_lab.training.stage40_policy_eval --action test --data data --root "$out" \
 > "$out/test_eval.log" 2>&1 || { echo BLOCKED_FINAL_TEST;tail -50 "$out/test_eval.log";exit 6; }
"$py" - "$out" <<'PY'
import json,pathlib,hashlib,sys
r=pathlib.Path(sys.argv[1]);s=json.loads((r/'dev_selection.json').read_text());x=json.loads((r/'final_test60_official.json').read_text())
assert x['status']=='measured' and x['dev25_disjoint'] and x['prior240_ifeval_keys_excluded']
assert x['test60_sha256']==s['untouched60_sha256']
assert x['dev_selection_sha256']==hashlib.sha256((r/'dev_selection.json').read_bytes()).hexdigest()
assert x['selected']==s['selected']
assert len(x['tested_labels'])>=2
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(r))+'\n'
 for p in sorted(r.rglob('*')) if p.is_file() and p.name!='SHA256SUMS' and not p.name.endswith('.log')))
print('STAGE40_FINAL60_PASS',x['selected'],x['selected_adapted'],x['paired_stratified_CIs_selected_minus'])
PY
