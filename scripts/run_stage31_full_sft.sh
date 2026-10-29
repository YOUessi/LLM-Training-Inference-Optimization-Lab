#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage31_approved_source_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_SOURCE_APPROVAL; exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE; exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK; exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_VLLM_LISTENER; exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY; exit 3; }
out=results/stage31_full512_two_epochs
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING_OUTPUT; exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/home/you/venvs/llm-opt-qlora/bin/python
for seed in 3111 3127;do
 for rank in 8 16;do
  name="rank${rank}_seed${seed}"
  echo "STAGE31_TRAIN_START $name" | tee -a "$out/progress.log"
  timeout 1600 "$python" -m llm_opt_lab.training.stage31_full_dolly \
   --rank "$rank" --seed "$seed" --input data --out "$out/${name}.json" \
   --adapter "$out/${name}_adapter" > "$out/${name}.log" 2>&1 || { echo "STAGE31_FAILED $name";tail -45 "$out/${name}.log"; exit 4; }
  "$python" - "$out/${name}.json" <<'PY'
import json,math,sys
x=json.load(open(sys.argv[1]));a=x['train_epochs']
assert x['status']=='measured' and x['train_unique_records']==512 and x['train_exposures']==1024
assert len(a)==2 and all(z['train_records_seen_unique']==512 for z in a)
assert x['selected_by_validation_epoch'] in (1,2)
assert x['best_validation_ce']<x['baseline_validation']['assistant_ce_nats']
assert math.isfinite(x['training_efficiency']['samples_per_second_excluding_first5'])
print('STAGE31_TRAIN_VALID',x['config']['rank'],x['config']['seed'],x['best_validation_ce'])
PY
 done
done
"$python" - "$out" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);cases=sorted(root.glob('rank*_seed*.json'))
assert len(cases)==4
payload={p.stem:json.loads(p.read_text()) for p in cases}
selected=min(payload,key=lambda name:payload[name]['best_validation_ce'])
best=payload[selected]
result={'status':'candidate_selected_from_validation_only','selected':selected,'rank':best['config']['rank'],
 'seed':best['config']['seed'],'best_epoch':best['selected_by_validation_epoch'],
 'val_ce':best['best_validation_ce'],'source_sha':(root/'source_sha').read_text().strip(),
 'all_validation_criteria':{k:x['best_validation_ce'] for k,x in payload.items()},
 'test_has_not_been_accessed':True}
(root/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
print('STAGE31_VALIDATION_WINNER',selected,result['val_ce'])
PY
read -r selected rank < <("$python" - "$out" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]+'/selection.json'))
print(x['selected'],x['rank'])
PY
)
"$python" -m llm_opt_lab.training.stage26_real_sft --action test --rank "$rank" --precision bf16 --steps 512 \
 --input data --adapter "$out/${selected}_adapter" --out "$out/selected_final_test.json" > "$out/final_test.log" 2>&1 || { tail -45 "$out/final_test.log";exit 5; }
"$python" - "$out" <<'PY'
import pathlib,json,hashlib,sys
r=pathlib.Path(sys.argv[1]);x=json.loads((r/'selected_final_test.json').read_text());
assert x['status']=='measured' and x['test_metrics']['count']==96
assert x['base_test_metrics']['count']==96
(r/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(r))+'\n' for p in sorted(r.rglob('*')) if p.is_file() and p.name not in ('SHA256SUMS',) and not p.name.endswith('.log')))
print('STAGE31_FROZEN_96_TEST_PASS','base_ce',x['base_test_metrics']['assistant_ce_nats'],'tuned_ce',x['test_metrics']['assistant_ce_nats'])
PY
