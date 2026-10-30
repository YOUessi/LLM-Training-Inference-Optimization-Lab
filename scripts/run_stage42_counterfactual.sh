#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
approved=$(cat results/stage42_approved_sha.txt 2>/dev/null || :)
[[ -n "$approved" && "$sha" == "$approved" ]] || { echo BLOCKED_SOURCE_APPROVAL;exit 3; }
[[ -z "$(git status --porcelain)" ]] || { echo BLOCKED_DIRTY_SOURCE;exit 3; }
exec 9>/home/you/.cache/llm_opt_shared_gpu.lock
flock -n 9 || { echo BLOCKED_GPU_LOCK;exit 3; }
! ss -ltn|grep -q ':8026 ' || { echo BLOCKED_VLLM_SERVER;exit 3; }
! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|grep -Eq '[0-9]' || { echo BLOCKED_GPU_IN_USE;exit 3; }
out=results/stage42_clean_router_96_128
[[ ! -e "$out" ]] || { echo BLOCKED_RESULT_OVERWRITE;exit 3; }
mkdir -p "$out";echo "$sha" > "$out/source_sha.txt"
export PYTHONPATH="$PWD/src:$PWD/third_party/ifeval/google_research" NLTK_DATA=/home/you/nltk_data HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Only one GPU run for one frozen router. Baseline responses are counterfactual audit, no test-based rerouting.
timeout 3000 /home/you/venvs/llm-opt-qlora/bin/python -m llm_opt_lab.training.stage42_counterfactual \
 --input data --adapter evidence/stage31_full512/selected_adapter \
 --out "$out/frozen_router_result.json" > "$out/generation.log" 2>&1 || {
 echo STAGE42_EXPERIMENT_FAILED
 tail -45 "$out/generation.log"
 exit 4
}
python3 - "$out" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);p=root/'frozen_router_result.json';x=json.loads(p.read_text())
assert x['status']=='measured' and len(x['real_ifeval_generations'])==96 and len(x['real_dolly_reference_ce'])==128
assert x['previous_source_identites_excluded']=={'IFEval':325,'Dolly':960}
assert x['router']['uses_official_labels_or_model_responses'] is False
assert sum(x['route_counts']['ifeval'].values())==96
assert sum(x['route_counts']['dolly'].values())==128
(root/'SHA256SUMS').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  frozen_router_result.json\n')
print('STAGE42_REAL_ROUTER_AUDIT_PASS',x['joint_gate_pass'],x['predeclared_joint_research_gate'])
PY
