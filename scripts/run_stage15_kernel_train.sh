#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"; out=results/stage15_kernel_train_20261029
[[ ! -e "$out" ]] || { echo ALREADY_EXECUTED;exit 3; }
mkdir -p "$out";git rev-parse HEAD > "$out/source_sha"
for shape in '128 1024' '256 4096';do
 read -r rows cols <<< "$shape"
 python3 -m llm_opt_lab.kernels.benchmark --rows "$rows" --cols "$cols" --warmup-ms 70 --rep-ms 160 --output "$out/kernel_${rows}x${cols}.json" | tee -a "$out/progress.log"
done
/home/you/conda_env/llm-opt-serving/bin/torchrun --standalone --nproc_per_node=1 -m llm_opt_lab.training.fsdp_smoke --steps 12 --output "$out/fsdp_world1.json" | tee -a "$out/progress.log"
