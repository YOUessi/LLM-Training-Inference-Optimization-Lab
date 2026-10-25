# vLLM service benchmark on Tang

This project does not modify system Python or previously configured model environments.

## Environment and model

- Dedicated venv: `$HOME/conda_env/llm-opt-serving`. It is created with `python3 -m venv --system-site-packages`: pip writes only to this venv, while it **reads the existing system/user PyTorch 2.10.0+cu128**. This is not a fully independent Torch installation. No old project files, weights or global packages are changed.
- Compatible installation target: `vllm==0.19.1`; **successful installation and runtime validation must be recorded before claiming the milestone**.
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

Or use the reproducible sequential concurrency sweep (includes two per-case warmup requests):

```bash
PYTHONPATH=src python3 -m llm_opt_lab.serving.sweep \
  --base-url http://127.0.0.1:8026 \
  --model qwen2.5-1.5b-instruct-local \
  --requests 24 --warmup-requests 2 --concurrency 1,2,4 \
  --max-tokens 128 --output-dir results/vllm_sweep_20261026
```

The sweep writes one raw JSON per concurrency setting and a derived summary,
aborting if a prior workload fails. Records keep the model and configuration;
they never store response text or API authentication headers.

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

## Compatibility decision

On Tang the existing Torch is `2.10.0+cu128`, Triton `3.6.0`, NVIDIA driver `580.178.04`. Public PyPI metadata says `vllm==0.19.1` requires `torch==2.10.0`, while `vllm==0.31.0` requires `torch==2.13.0` and pulls newer CUDA 13 packages. An unfinished 0.31 download was aborted and its *newly created* environment replaced.

The pinned 0.19.1 environment still requires end-to-end import/server validation; the version choice alone does **not** prove functionality.

## Network downloads and verified wheels

If pip's default 15-second read timeout interrupts the large vLLM wheel download:

```bash
bash scripts/download_vllm_wheel.sh
# After SHA256_OK, install from the verified local wheel:
"$HOME/conda_env/llm-opt-serving/bin/python" -m pip install \
  --timeout 180 --retries 6 \
  "results/vendor/vllm-0.19.1-cp38-abi3-manylinux_2_31_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

The pinned wheel SHA-256 is `71a87f46cafab4489c69a5c5c83b870d0235e5694d8222303d460576293dc719`, read from PyPI's 0.19.1 release metadata. The existing system Torch 2.10.0+cu128 is read-only to the venv; this is not a claim that all environment dependencies are independently isolated. If network installation fails, report BLOCKED, not a successful serving experiment.
