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

## Offline PyTorch fallback (NOT vLLM)

When PyPI wheel downloads are blocked by TLS timeouts or severe throttling, do **not**
report a successful vLLM deployment. Tang already has cached
`Qwen2.5-1.5B-Instruct`, PyTorch and Transformers, allowing a separate
**real-model generation baseline**:

```bash
PYTHONPATH=src python3 -m llm_opt_lab.serving.hf_baseline \
  --batch-sizes 1,2,4 --repeats 3 --max-new-tokens 32 \
  --output results/hf_offline_baseline.json
```

The baseline reports full generation wall time, true emitted token counts
(until EOS), output-token throughput, request throughput and peak GPU memory.
The timing intentionally excludes tokenization and captures prefill + decode
together. **TTFT and TPOT remain null.** It is not a server,
has no dynamic batching/scheduler, and should never be compared to vLLM
as if the concurrency semantics were identical.

### Offline baseline Pillow compatibility

The global Tang Python has Pillow 9.0.1, incompatible with Transformers 4.57.6's `PIL.Image.Resampling`. Use the lab-created venv which reuses Torch 2.10 read-only and install `Pillow==11.3.0` **inside the venv only**. Run the offline baseline through `$HOME/conda_env/llm-opt-serving/bin/python`; do not install Pillow globally.

## Confirmed real deployment — 2026-10-28

The previously blocked \`vllm==0.19.1\` Linux wheel is now downloaded and installed on Tang in the dedicated \`llm-opt-serving\` virtualenv (which still reads the existing Torch 2.10.0+cu128). PyPI official wheel SHA-256 was verified:

\`71a87f46cafab4489c69a5c5c83b870d0235e5694d8222303d460576293dc719\`

The initial 8-request × 1/2/4-concurrency vLLM stream sweep succeeded on
Qwen2.5-1.5B-Instruct. For stronger workload/sequence validation, run:

\`\`\`bash
bash scripts/run_vllm_benchmark.sh
\`\`\`

This bounds GPU startup, refuses occupied devices and occupied ports, executes
three sequential workload sweeps with warmup, writes results under ignored
\`results/\`, and terminates only the server process it owns.
It refuses overwriting an existing result directory. 200+ pre-existing
GPU jobs are not terminated or modified.

Critical provenance: serving endpoint binds to \`127.0.0.1:8026\`, NVIDIA RTX
4090 Laptop, model snapshot
\`989aa7980e4cf806f80c7fef2b1adb7bc71aa306\`,
\`--gpu-memory-utilization 0.50\`, \`--max-model-len 2048\`,
\`--max-num-seqs 8\`, FP16, vLLM 0.19.1. Different workload shapes and run
orders are reported separately to prevent conflating speedups.

An unrelated \`mediapipe\` system-site package expects older Protobuf than
the protobuf installed inside the lab venv. **No global dependency was
changed to silence this conflict.** Only verify vLLM's own import/runtime path.
\n### Correcting the long-output protocol\n\nThe original \`long_decode\` run (max_tokens=128) actually generated exactly **21 completion tokens per request**. It is preserved as an invalid long-generation stress test. The corrected runner specifies a long-answer instruction and gates \`mean(actual completion_tokens)>=90\` before marking that sweep measured. Source code changes and raw results must be pinned to the matching commit.\n