#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v nsys >/dev/null 2>&1; then
  echo "BLOCKED: nsys not found. Install Nsight Systems on NVIDIA GPU host." >&2
  exit 2
fi
if ! command -v python >/dev/null 2>&1; then
  echo "BLOCKED: activate the dedicated GPU Python environment." >&2
  exit 2
fi
mkdir -p results
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
prefix="results/nsys_kernel_${stamp}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --output "$prefix" \
  python -m llm_opt_lab.kernels.benchmark --output "${prefix}.json" "$@"
echo "Nsight Systems saved: ${prefix}.nsys-rep"
