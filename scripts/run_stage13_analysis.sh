#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=results/stage13_full_analysis.json
if [[ -e "$out" ]];then
 echo "BLOCKED: prior Stage13 analysis exists and may not be overwritten" >&2;exit 3
fi
source_sha="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]];then
 echo "BLOCKED: source tree dirty before evidence analysis" >&2;exit 3
fi
export PYTHONPATH="$PWD/src"
python3 -m llm_opt_lab.serving.stage13_analysis \
 --raw-dir results \
 --expected-source-commit "$source_sha" \
 --out "$out"
echo "STAGE13_FROZEN_ANALYSIS_PASS $source_sha"
