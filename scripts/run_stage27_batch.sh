#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage27_approved_source_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_UNAPPROVED_SOURCE;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_SERVING_RUNNING;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_BUSY;exit 3; }
out=results/stage27_qwen3_real_kernels
[[ ! -e "$out" ]] || { echo BLOCKED_NO_OVERWRITE;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 -m llm_opt_lab.kernels.stage27_benchmark --warmup-ms 90 --rep-ms 200 --profile --out "$out/benchmark.json" > "$out/benchmark.log" 2>&1 || { tail -65 "$out/benchmark.log";exit 4; }
python3 - "$out/benchmark.json" <<'PY'
import sys,json,math
x=json.load(open(sys.argv[1]));assert x['status']=='measured' and len(x['cases'])==8
for r in x['cases']:
 assert all(math.isfinite(r[k]) for k in ('numerical_max_abs_error','speedup_vs_compile','speedup_vs_eager'))
 assert r['speedup_vs_compile']>0
print('STAGE27_BENCH_VALIDATED',len(x['cases']),'cases',len(x['profiler']),'profiles')
PY
(cd "$out" && sha256sum benchmark.json > SHA256SUMS)
