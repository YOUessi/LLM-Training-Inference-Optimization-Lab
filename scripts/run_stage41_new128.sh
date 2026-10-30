#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage41_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_SOURCE_SHA;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_SERVER_RUNNING;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_USED;exit 3; }
out=results/stage41_fresh128_new_models
[[ ! -e "$out" ]] || { echo BLOCKED_RESULTS_ALREADY_EXIST;exit 3; }
mkdir -p "$out"
echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
stage38=/home/you/projects/llmopt_stage38_frozen/results/stage38_guarded_alpha_ifeval
stage40=/home/you/projects/llmopt_stage40_v2_frozen/results/stage40_real_constraint_rehearsal
if [[ ! -f "$stage38/final_untouched101_test.json" || ! -f "$stage40/final_test60_official.json" ]];then
 echo BLOCKED_PRIOR_INDEPENDENT_TESTS_NOT_COMPLETE
 exit 4
fi
timeout 1000 /home/you/venvs/llm-opt-qlora/bin/python -m llm_opt_lab.training.stage41_new128_tradeoff \
 --data data --stage38 "$stage38" --stage40 "$stage40" \
 --out "$out/fresh128_four_conditions.json" > "$out/progress.log" 2>&1 || {
 echo STAGE41_FAILED
 tail -30 "$out/progress.log"
 exit 5
}
python3 - "$out" <<'PY'
import json,pathlib,hashlib,sys
folder=pathlib.Path(sys.argv[1]);p=folder/'fresh128_four_conditions.json'
x=json.loads(p.read_text())
assert x['status']=='measured' and x['excluded_previous832_identites']
assert x['counts']=={'examples':128,'categories':8,'models':4}
for name in ('base','original_full','guarded_strength','format_replay'):
 assert x['overall'][name]['examples']==128
(folder/'SHA256SUMS').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  fresh128_four_conditions.json\n')
print('STAGE41_VERIFIED_UNSEEN_DOLLY128',x['overall'])
PY
