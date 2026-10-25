#!/usr/bin/env bash
set -euo pipefail

# Start vLLM on Tang only in a dedicated virtualenv. Never listen on 0.0.0.0.
ENV_ROOT="${VLLM_ENV_ROOT:-$HOME/conda_env/llm-opt-serving}"
if [[ ! -x "$ENV_ROOT/bin/vllm" ]]; then
  echo "BLOCKED: vLLM CLI missing from isolated env: $ENV_ROOT" >&2
  exit 2
fi

SNAPSHOTS="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
if [[ ! -d "$SNAPSHOTS" ]]; then
  echo "BLOCKED: offline Qwen2.5-1.5B-Instruct cache missing" >&2
  exit 2
fi
MODEL_DIR="${VLLM_MODEL_DIR:-$(find "$SNAPSHOTS" -mindepth 1 -maxdepth 1 -type d | sort | head -1)}"
if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/model.safetensors" ]]; then
  echo "BLOCKED: offline model weights/config unavailable in $MODEL_DIR" >&2
  exit 2
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${VLLM_PORT:-8026}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
echo "Starting loopback-only vLLM, offline model: $(basename "$MODEL_DIR")"
echo "Port: 127.0.0.1:$PORT, GPU memory target: $GPU_MEMORY_UTILIZATION"
exec "$ENV_ROOT/bin/vllm" serve "$MODEL_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --dtype float16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --served-model-name qwen2.5-1.5b-instruct-local
