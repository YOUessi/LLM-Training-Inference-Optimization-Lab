#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v ncu >/dev/null 2>&1; then
  echo "BLOCKED: ncu not found. Install Nsight Compute on NVIDIA GPU host." >&2
  exit 2
fi
if ! command -v python >/dev/null 2>&1; then
  echo "BLOCKED: activate the dedicated GPU Python environment." >&2
  exit 2
fi
mkdir -p results
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
prefix="results/ncu_kernel_${stamp}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Profile at most 10 launches. Instrumented runtime is NOT native speedup evidence.
ncu --set basic --launch-count 10 --target-processes all \
  --force-overwrite --export "$prefix" \
  python -m llm_opt_lab.kernels.benchmark --output "${prefix}.json" "$@"
echo "Nsight Compute saved: ${prefix}.ncu-rep"
