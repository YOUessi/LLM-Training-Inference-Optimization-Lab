# vLLM service benchmark on Tang

This project does not modify system Python or previously configured model environments.

## Environment and model

- Dedicated venv: `$HOME/conda_env/llm-opt-serving` (separate from CodeRL-Lab, Isaac Lab, other Tang projects).
- Tested installation target: `vllm==0.31.0`; **successful installation and runtime validation must be recorded before claiming the milestone**.
- Existing offline model: Qwen2.5-1.5B-Instruct Hugging Face cache on Tang. No need to download model weights.
- Service only listens on `127.0.0.1:8026`; no API keys or public access required.
- `CUDA_VISIBLE_DEVICES=0`, `--gpu-memory-utilization 0.50`, `--max-model-len 2048`; never start if GPU is busy with other jobs.

## Run

```bash
bash scripts/run_vllm_local.sh
```

In a separate Tang shell:

```bash
cd "$HOME/projects/LLM-Training-Inference-Optimization-Lab"
PYTHONPATH=src python3 -m llm_opt_lab.serving.bench_client \
  --base-url http://127.0.0.1:8026 --model qwen2.5-1.5b-instruct-local \
  --requests 24 --concurrency 1 --warmup-requests 2 --max-tokens 128 \
  --output results/vllm_c1.json

# Repeat using the identical request distribution at concurrency 2 and 4.
```

Environment and metrics are committed only after actual successful service requests.
`completion_tokens` must come from provider usage; missing usage => TPOT and tokens/sec are null.
Track model revision, PyTorch/CUDA/vLLM versions and hardware for every run.
Serving results are distinct from RMSNorm microbenchmarks.

## Safe teardown

Stop the foreground vLLM process after benchmarking. Verify port 8026 is closed
and GPU allocations are released. Never stop preexisting Tang processes.

## Current limitations

This lab's client is an OpenAI-compatible SSE harness; a successful mocked
HTTP integration test is **not** evidence of vLLM server throughput. Tokens/sec
and P95 TTFT need an actually loaded model and requests on Tang.
