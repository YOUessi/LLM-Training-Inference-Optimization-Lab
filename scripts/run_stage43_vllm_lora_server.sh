#!/usr/bin/env bash
set -euo pipefail
# Isolated loopback-only vLLM Qwen3-8B-AWQ dual-base/LoRA integration smoke.
ENV_ROOT="${VLLM_ENV_ROOT:-/home/you/conda_env/llm-opt-serving}"
[[ -x "$ENV_ROOT/bin/vllm" ]] || { echo BLOCKED_VLLM_ENV;exit 3; }
MODEL=/home/you/models/Qwen3-8B-AWQ
[[ -f "$MODEL/config.json" && -f "$MODEL/model.safetensors.index.json" ]] || { echo BLOCKED_AWQ_MISSING;exit 3; }
REPO=/home/you/projects/LLM-Training-Inference-Optimization-Lab
LORA="$REPO/evidence/stage31_full512/selected_adapter"
[[ -f "$LORA/adapter_config.json" && -f "$LORA/adapter_model.safetensors" ]] || { echo BLOCKED_LORA_MISSING;exit 3; }
python3 - "$LORA/adapter_config.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['r']==16 and set(x['target_modules'])=={'q_proj','v_proj'}
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
exec "$ENV_ROOT/bin/vllm" serve "$MODEL" \
 --host 127.0.0.1 --port 8033 \
 --dtype float16 --max-model-len 2048 --max-num-seqs 16 \
 --gpu-memory-utilization .82 --no-enable-prefix-caching \
 --served-model-name qwen3-8b-awq-local \
 --enable-lora --max-lora-rank 16 --max-loras 1 --lora-dtype float16 \
 --lora-modules "qwen3-8b-dolly-r16=$LORA"
