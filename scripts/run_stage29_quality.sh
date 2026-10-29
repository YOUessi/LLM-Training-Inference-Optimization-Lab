#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD); approved=$(cat results/stage29_approved_source_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$approved" == "$sha" ]] || { echo BLOCKED_UNAPPROVED_SOURCE;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn | grep -q ':8026 ' || { echo BLOCKED_LIVE_SERVER;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY;exit 3; }
out=results/stage29_frozen_quality
[[ ! -e "$out" ]] || { echo BLOCKED_EXISTING_OUTPUT;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/you/venvs/llm-opt-qlora/bin/python -m llm_opt_lab.training.stage29_instruction_quality \
 --adapter evidence/stage26_sft_v2/selected_B_adapter --input data --limit 96 \
 --out "$out/paired_generated_quality.json" > "$out/progress.log" 2>&1 || {tail -40 "$out/progress.log";exit 4;}
python3 - "$out/paired_generated_quality.json" <<'PY'
import sys,json
x=json.load(open(sys.argv[1]));assert x['status']=='measured' and x['total_cases']==96 and len(x['cases'])==96
print('STAGE29_PAIRED_96_SUCCESS',x['overall'])
PY
(cd "$out" && sha256sum paired_generated_quality.json > SHA256SUMS)
