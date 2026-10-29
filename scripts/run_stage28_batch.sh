#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage28_approved_source_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_UNAPPROVED_SOURCE;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_SERVING;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY;exit 3; }
out=results/stage28_real_transformer_fullstate
[[ ! -e "$out" ]] || { echo BLOCKED_NO_OVERWRITE;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/you/conda_env/llm-opt-serving/bin/python -m llm_opt_lab.training.stage28_recovery_bench \
 --out "$out/comparison.json" > "$out/comparison.log" 2>&1 || { tail -60 "$out/comparison.log";exit 4; }
python3 - "$out/comparison.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['status']=='pass' and x['exact_final_adapter_and_optimizer'] and len(x['runs'])==4
assert x['runs'][-1]['restarted_from_step']==6
print('STAGE28_RESTART_VERIFIED',len(x['runs']),'runs',len(x['checkpoints_verified']),'raw_checkpoint_files')
PY
(cd "$out" && sha256sum comparison.json > SHA256SUMS)
