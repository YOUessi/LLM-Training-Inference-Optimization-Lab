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

- [Sequential vLLM concurrency sweeps](src/llm_opt_lab/serving/sweep.py) avoid overlapping loads; warmup and provider token usage are tracked.

- [Pinned resumable vLLM wheel download](scripts/download_vllm_wheel.sh) (SHA-256 verified, Tang only).

- [Offline Qwen2.5 generation benchmark (not vLLM)](src/llm_opt_lab/serving/hf_baseline.py): fallback when network wheel downloads fail, logs real GPU wall time and peak allocated memory.

- [Measured 1.5B offline Qwen inference results](docs/RESULTS_2026-10-26.md#offline-real-model-inference-baseline-not-vllm). This is deliberately distinguished from vLLM Serving.

## Stage 2: exact restart and real SSE throughput (2026-10-26)

[Full Stage 2 technical report](docs/STAGE2_REPORT_2026-10-26.md). Real CPU/GPU DDP Checkpoint V2 exact-resume validation passed and a loopback-only **Transformers** streaming service produced TTFT, client-estimated TPOT and output token throughput under 1/2/4 request concurrency. These are **not vLLM benchmarks**. Raw machine-generated evidence is under `docs/evidence/stage2_*` with a source-commit manifest.

### Training infrastructure extension

Real [single-GPU FSDP1 smoke](src/llm_opt_lab/training/fsdp_smoke.py) exercised 20 forward/backward/optimizer steps. World-size=1 selects NO_SHARD, not genuine distributed sharding. See [Stage 2 evidence](docs/STAGE2_REPORT_2026-10-26.md#4-real-pytorch-fsdp1-single-gpu-execution-no_shard).

### Real vLLM deployment (October 28, 2026)

vLLM **0.19.1 is installed and actually ran** on Tang with locally cached
Qwen2.5-1.5B-Instruct. First real SSE measurements are complete;
stronger multi-order workload experiments are in
[\`scripts/run_vllm_benchmark.sh\`](scripts/run_vllm_benchmark.sh).
The vLLM serving results must not be confused with the earlier serialized
Transformers baseline. See [server runbook](docs/VLLM_RUNBOOK.md).

### Stage 3: real vLLM measurements validated

[Stage 3: real SSE TTFT/TPOT, 1/2/4/8 concurrent clients and corrected 128-token long decode](docs/STAGE3_REPORT_2026-10-28.md). Source experiment raw JSON and negative results are archived under docs/evidence/stage3_vllm_repeated/ and docs/evidence/stage3_vllm_invalid_long/. No Nsight or multi-GPU/FSDP sharding results are claimed.

### Nsight Systems real CUDA traces

[Stage 3 Nsight Systems kernel profiling report](docs/STAGE3_NSYS_REPORT_2026-10-28.md) includes verified CUDA kernel launches, stats CSV, exact raw trace SHA256 and permission limits (no hardware-counter/occupancy claim).\n
- [Targeted NVTX RMSNorm profiling and exact kernel-launch counts](docs/STAGE3_NVTX_REPORT_2026-10-28.md) (30 calls/variant; hardware counter limits disclosed).

### Stage 4: Prefix caching experimental design

See [frozen experiment protocol](docs/PREFIX_CACHE_PROTOCOL.md) and [cache on/off runner](scripts/run_prefix_cache_ablation.sh). Shared/distinct long prefixes, explicit cache enabled/disabled, two seeds and concurrency 1/8. Real prompt_tokens are used to enforce valid workloads. Protocol is not a result.

### Stage 4: measured Prefix Cache causal-control benchmark

[Stage 4 detailed results](docs/STAGE4_PREFIX_CACHE_REPORT_2026-10-28.md) cover 384 actual SSE requests on Qwen2.5-1.5B in vLLM 0.19.1, both cache states, shared vs distinct-prefix negative controls, two seeds and both 1/8 concurrency. Results report actual provider prompt tokens and vLLM Prometheus token-hit counters; an invalid first experiment is retained as negative process evidence. This is a controlled application-level cache ablation, not a new KV-cache implementation.

### Stage 5: KV cache capacity & mixed-context prefill

[Stage5 frozen protocol](docs/STAGE5_PROTOCOL.md), [bounded cache eviction probe](src/llm_opt_lab/serving/kv_capacity_probe.py), [mixed-context SSE workload](src/llm_opt_lab/serving/mixed_context_bench.py).

### Recurring repository status (every 30 minutes, best effort)

[GitHub Actions plan audit](.github/workflows/plan-audit.yml) runs CPU regression/integrity checks and compares committed evidence against the Roadmap, posting next-action guidance in each Action run summary. **GitHub Actions cannot use Tang GPU, cannot edit work automatically, and cannot deliver half-hour messages to this ChatGPT thread**. Live GPU milestones require a separately authorized online Tang executor.

### Stage 5 — real CUDA KV eviction and mixed-context serving results

[Real 256-block KV eviction evidence](docs/STAGE5_KV_CAPACITY_REPORT_2026-10-28.md) and [192-request heterogeneous input prefill/throughput](docs/STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md) on Qwen2.5-1.5B FP16/Tang RTX4090 Laptop GPU. Raw 9 JSON files and SHA-256 manifest under docs/evidence/stage5_kv_and_mixed/. Stage6 naturalistic mixed-load and multiple GPU block budgets remain open.

### Stage6 initial result: latency SLO/throughput frontier

[Conservative two-seed empirical P95 TTFT SLO selection](docs/STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md) from 192 real Stage5 SSE requests. At experimental 150ms P95 TTFT target concurrency 2 (min 207 output tok/s) qualifies; at 250ms concurrency 4 (min 320.8 tok/s); see report for scope limits. This is derived analysis, **not a fresh GPU run**.

### Stage6 independent capacity-retention tests

[Four-point KV retention matrix protocol](docs/STAGE6_CAPACITY_MATRIX_PROTOCOL.md), [reproducible per-case clean-server runner](scripts/run_stage6_capacity_matrix.sh), and [classifier](src/llm_opt_lab/serving/kv_capacity_curve.py). A valid retained cache counts as a measured outcome, not a false failure. Run only on an idle GPU.

### Stage6 measured retention-vs-capacity

[Four independent real vLLM KV capacity points](docs/STAGE6_CAPACITY_MATRIX_REPORT_2026-10-28.md): 128 and 256 Block instances lose the anchor after four unrelated long contexts; 512 retains it after four but loses it after twelve. Each condition used a fresh model server and real prefix hit counters; no claimed exact general eviction threshold.

### Stage7 frozen multi-genre natural prose workload

[Public-domain-in-USA 30-excerpt corpus](data/README.md), [preregistered GPU experiment](docs/STAGE7_PROTOCOL.md), [vLLM mixed prefill/decode runner](scripts/run_stage7_natural.sh). Four actual input lengths and two generated length instructions are validated by real provider usage, not max-token caps. **Stage7 performance conclusions are pending the real Tang GPU run.**

### Stage7 — natural prose distribution-shift benchmark (real GPU)

[Full Stage7 report](docs/STAGE7_REPORT_2026-10-28.md) and [raw 384-request case evidence](docs/evidence/stage7_natural). Thirty attributed nonoverlapping excerpts from five classical prose genres replaced repeated synthetic paragraphs. Compared two decoding budgets (actual ~31 vs 128 output tokens), four real input lengths, 1/2/4/8 concurrency, two seeds. A protocol-invalid first run was retained and corrected **without widening the original input-token bounds**.

Stage8 still requires more modern, mixed-linguistic domains, true natural output stopping and replayable arrival distributions.
