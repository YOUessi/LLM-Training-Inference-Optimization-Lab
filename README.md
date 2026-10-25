# LLM Training & Inference Optimization Lab

An independent, reproducible systems engineering lab for **LLM inference serving**, **CUDA/Triton kernels**, **GPU profiling**, and **distributed training**. Targeted at AI Infra, LLM serving, and training-framework internship roles.

> **Status (2026-10-26): scaffold and CPU tests.** A kernel benchmark only becomes a measured result after execution on a real NVIDIA GPU. No speedups, multi-GPU scaling, or deployed-serving metrics are claimed before experiments run.

## Why a new repository?

Existing work is valuable but answers other questions:
- [agentic-gpu-debugger](https://github.com/YOUessi/agentic-gpu-debugger): functional correctness, Compute Sanitizer and evidence-grounded CUDA fault repair; **not a kernel performance optimization platform**.
- [CodeRL-Lab](https://github.com/YOUessi/CodeRL-Lab): model post-training and verified-reward experiments; **not itself a general inference/serving optimization laboratory**.

This repo uses the former's *correctness-before-performance* discipline and the latter's *pinned-config/fair-comparison* discipline. Initial implementation is original; source files from either repo have **not** been copied. See [docs/REUSE.md](docs/REUSE.md) for pinned upstream references and boundaries.

## Workstreams

| Track | Baseline | Optimization | Measured outcomes |
|---|---|---|---|
| GPU kernel | PyTorch RMSNorm/Softmax | Triton implementations | correctness, latency, speedup across shapes/dtypes |
| Serving | OpenAI-compatible vLLM endpoint | concurrency / request-mix sweeps | TTFT, TPOT (if server returns token usage), completion throughput, error rate |
| Training | synthetic DDP training loop | profiling & future FSDP studies | per-step time, peak memory, scaling, reproducible checkpoint |
| Profiling | Nsight Systems / Nsight Compute | measure and identify bottlenecks | traced workload, tool versions, raw reports |

## Quick start (Mac / any machine, standard library only)

```bash
cd LLM-Training-Inference-Optimization-Lab
PYTHONPATH=src python3.12 -m unittest discover -s tests -v
```

CPU tests do not fake CUDA performance results. Optional NVIDIA requirements are needed only on Tang/Linux GPU.

## GPU kernel experiments (Linux/NVIDIA CUDA + torch + Triton required)

Install PyTorch for the actual CUDA/driver combination, then Triton separately. Do **not** install arbitrary CUDA wheels on macOS. Example after appropriate environment provisioning:

```bash
PYTHONPATH=src python -m llm_opt_lab.kernels.benchmark --rows 128 --cols 1024 --dtype float16 --output results/kernel_128x1024.json
PYTHONPATH=src python -m llm_opt_lab.kernels.benchmark --rows 256 --cols 4096 --dtype float32 --output results/kernel_256x4096.json
```

The program exits nonzero if no CUDA device is available, tests precision vs PyTorch, and only writes results after completing the requested measurements. Timing is in milliseconds.
For a stronger fused baseline, add `--torch-compile`: steady-state Torch Inductor (`torch.compile`) and custom Triton performance will be reported **separately**. Compilation time is excluded; comparisons use the same tensor and output dtype.


## Serving experiments (OpenAI-compatible vLLM endpoint)

Run your own vLLM server **separately** on a GPU host. Then:

```bash
PYTHONPATH=src python -m llm_opt_lab.serving.bench_client \
  --base-url http://127.0.0.1:8000 --model YOUR_LOADED_MODEL \
  --requests 24 --concurrency 4 --max-tokens 128 \
  --output results/serving_c4.json
```

The benchmark counts **TTFT from first nonempty streamed content** and uses provider-reported completion token usage for TPOT. If token usage is absent, TPOT and true token-throughput remain `null` — SSE chunks are not assumed to equal tokens. Do not expose unauthenticated vLLM servers publicly.

## Distributed training (torchrun, real DDP)

```bash
# 1 CPU worker is a smoke test, NOT multi-GPU scaling evidence
PYTHONPATH=src torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  -m llm_opt_lab.training.ddp_train --steps 5 --device cpu \
  --output results/ddp_cpu_smoke.json

# Requires 2 visible NVIDIA GPUs
PYTHONPATH=src torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m llm_opt_lab.training.ddp_train --steps 100 --device cuda \
  --output results/ddp_cuda_2gpu.json
```

This is a deliberately small synthetic training benchmark to validate the distributed contract and timing. **It is not evidence that an LLM was trained with FSDP.** FSDP experiments are separate milestones, not a label applied to DDP.

## Profiling

```bash
bash scripts/profile_nsys.sh --rows 256 --cols 4096
bash scripts/profile_ncu.sh --rows 256 --cols 4096
```

Scripts fail if Nsight tools or CUDA runtime are unavailable; trace artifacts stay local under ignored `results/`. Read [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) before comparisons.

## Engineering constraints

- **Repository as source of truth:** develop on Mac, track source commits in GitHub; run Tang only for CUDA, vLLM, or multi-GPU measurements.
- Always capture toolchain/GPU/commit and **every failed experiment**.
- Torch baseline and Triton workload must match input shape, dtype, semantics and tolerance.
- Do not turn smoke-test output into claimed measured speedups.
- Avoid modifying the two upstream projects; link or explicitly vendor pinned pieces only after a review.
- Logs and results with tokens, credentials or private inputs must not be committed.

See [GPU measurements and corrected fused-baseline comparisons](docs/RESULTS_2026-10-26.md), [docs/ISSUES_SOLUTIONS.md](docs/ISSUES_SOLUTIONS.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

### RMSNorm configuration search

See [experimental tuning and independent validation](docs/RMSNORM_TUNING_RESULTS.md): grouped-row scheduling did not produce general improvement; do not use it as a default kernel policy.

### Deployment and profiling

- [Loopback-only vLLM server and workload procedure](docs/VLLM_RUNBOOK.md).
- [Experimental Torch profiler](src/llm_opt_lab/kernels/profile_rmsnorm.py) while Nsight hardware counters are unavailable.
