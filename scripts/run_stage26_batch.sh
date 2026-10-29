#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source_sha="$(git rev-parse HEAD)"
approved="$(cat results/stage26_approved_source_sha.txt 2>/dev/null || :)"
[[ -n "$approved" && "$source_sha" == "$approved" ]] || { echo BLOCKED_APPROVAL_SHA;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_MODEL_SERVER_RUNNING;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY;exit 3; }
out=results/stage26_dolly_qlora_v1
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING_OUTPUT;exit 3; }
mkdir -p "$out"
printf '%s\n' "$source_sha" > "$out/source_sha"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python=/home/you/venvs/llm-opt-qlora/bin/python
run_variant(){
 local name="$1" rank="$2" micro="$3" accum="$4" precision="$5" gc="$6"
 local args=(--action train --rank "$rank" --micro-batch "$micro" --accum "$accum" --precision "$precision" --steps 96 --seed 2610 --input data --adapter "$out/${name}_adapter" --out "$out/${name}.json")
 if [[ "$gc" == yes ]];then args+=(--grad-checkpoint);fi
 echo "SFT_START $name rank=$rank micro=$micro accum=$accum precision=$precision gc=$gc" | tee -a "$out/progress.log"
 timeout 1200 "$python" -m llm_opt_lab.training.stage26_real_sft "${args[@]}" > "$out/${name}.log" 2>&1 || {
   echo "SFT_FAILED $name";tail -35 "$out/${name}.log";exit 4;
 }
 "$python" - "$out/${name}.json" <<'PY'
import json,sys,math
x=json.load(open(sys.argv[1]));a=x['validation_before'];b=x['validation_after'];c=x['training']
assert x['status']=='measured' and a['count']==b['count']==96 and c['measured_steps']==93
assert all(math.isfinite(z) for z in (a['assistant_ce_nats'],b['assistant_ce_nats'],c['median_step_ms'],c['peak_allocated_gib']))
print('SFT_VALID',x['config'],'VAL_BEFORE',a['assistant_ce_nats'],'VAL_AFTER',b['assistant_ce_nats'],'TPS',c['sample_throughput_per_second'])
PY
}
run_variant A 8 1 2 bf16 no
run_variant B 16 1 2 bf16 no
run_variant C 8 2 1 bf16 no
run_variant D 8 1 2 bf16 yes
run_variant E 8 1 2 fp16 no
"$python" - "$out" <<'PY'
import json,sys,pathlib,math
root=pathlib.Path(sys.argv[1]);names='ABCDE';d={n:json.loads((root/(n+'.json')).read_text()) for n in names}
valid={n:r for n,r in d.items() if r['validation_after']['assistant_ce_nats']<r['validation_before']['assistant_ce_nats']}
summary={'status':'comparison','metrics':{n:{'val_ce':r['validation_after']['assistant_ce_nats'],'base_val_ce':r['validation_before']['assistant_ce_nats'],
 'tokens_per_second':r['training']['sample_throughput_per_second'],'peak_gib':r['training']['peak_allocated_gib']} for n,r in d.items()},'passed_initial_quality_gate':list(valid)}
if not valid:raise ValueError('No candidate improves heldout validation CE')
selected=min(valid,key=lambda n:valid[n]['validation_after']['assistant_ce_nats'])
summary['selected']=selected
(root/'selection.json').write_text(json.dumps(summary,indent=2)+'\n')
print('SFT_SELECTION',selected,'val_CE',valid[selected]['validation_after']['assistant_ce_nats'])
PY
selected=$("$python" - "$out" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]+'/selection.json'))['selected'])
PY
)
"$python" -m llm_opt_lab.training.stage26_real_sft --action test --steps 96 --input data --adapter "$out/${selected}_adapter" --out "$out/selected_test.json" > "$out/test.log" 2>&1 || { tail -30 "$out/test.log";exit 5; }
"$python" - "$out" <<'PY'
import json,pathlib,sys,hashlib
root=pathlib.Path(sys.argv[1]);files=sorted(root.glob('*.json'))
assert len(files)==7, len(files)
a=json.loads((root/'selected_test.json').read_text());assert a['status']=='measured' and a['test_metrics']['count']==96
(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(root))+'\n' for f in sorted(root.rglob('*')) if f.is_file() and not f.name.endswith('.log') and f.name!='SHA256SUMS'))
print('STAGE26_COMPLETE',json.loads((root/'selection.json').read_text())['selected'],'TEST',a['test_metrics'])
PY
