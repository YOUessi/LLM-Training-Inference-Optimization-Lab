# Stage 2 — DDP exact restart & real streaming serving

**Experiment date:** 2026-10-26. All numerical claims in this document are derived from committed machine-generated evidence. **This is not vLLM or multi-GPU/FSDP training evidence.**

## 1. DDP exact checkpoint recovery (real `torchrun`)

With fixed seed 2026, batch size 16, a synthetic MLP (32 → 64 → 32), and total 8 steps, we executed independently:

- Uninterrupted 8 steps;
- 3 steps and checkpoint;
- A new `torchrun` process resumed the checkpoint for 5 more steps.

**Both CPU and single-GPU CUDA: PASS.** The uninterrupted/restarted model weights, complete optimizer state, all per-rank RNG states, and SHA-256 of the model bytes matched **exactly**.

| Device | World Size | Model | Optimizer | Per-rank RNG | Model SHA-256 equality |
|---|---:|---|---|---|---|
| CPU | 1 | equal | equal | equal | equal |
| CUDA (RTX 4090 Laptop) | 1 | equal | equal | equal | equal |

Artifacts: [CPU validation](evidence/stage2_ddp_cpu_resume.json), [CUDA validation](evidence/stage2_ddp_cuda_resume.json). Source training implementation: `1105581`; checkpoint format V2.

### Failure → correction

The previous checkpoint implementation omitted the training data generator RNG state. Before correction, CPU uninterrupted and resumed models differed by up to **0.00487** in a first-layer weight. After adding per-rank RNG state, the first CUDA resume run failed because tensors loaded directly onto CUDA were passed to `Generator.set_state()`, which requires CPU ByteTensors. Loading checkpoint metadata on CPU resolved that mismatch. The resulting exact checks now pass.

**Boundary:** a single NVIDIA GPU does not validate multi-GPU scaling, communication overlap, sharded optimizer restart, or FSDP.

## 2. Real streamed LLM server (Transformers, NOT vLLM)

Deployed local **Qwen2.5-1.5B-Instruct** in FP16 on Tang; actual model decoding used `TextIteratorStreamer` and an OpenAI-compatible SSE loopback-only endpoint. The server explicitly serialized generation with a lock, with **no continuous/dynamic batching**.

Fixed 8 measured requests and 1 warmup per concurrency, output cap 32 tokens. Results from the corrected `openai_compatible_sse_concurrency_sweep` with `engine_label=transformers_serialized`:

| Concurrent requests | P95 TTFT | P95 client-estimated TPOT | P95 total latency | Completion tokens/s | Successful |
|---|---:|---:|---:|---:|---:|
| 1 | 39.4 ms | 8.70 ms | 209.7 ms | 104.3 | 8/8 |
| 2 | 224.0 ms | 8.72 ms | 393.2 ms | 108.6 | 8/8 |
| 4 | 601.5 ms | 8.53 ms | 770.2 ms | 109.5 | 8/8 |

All 24 measured requests succeeded and their provider-reported completion-token usage coverage is **100%**.

**Finding:** as concurrency increased from 1 to 4, TTFT became much worse (queueing), while total output-token throughput only changed slightly. This confirms the serial scheduler is the bottleneck in this experimental baseline. Client TPOT includes stream finishing/network overhead and is *not* a kernel-per-token latency measurement.

Artifacts: [sweep summary](evidence/stage2_stream_summary.json), [concurrency=1](evidence/stage2_stream_c1.json), [concurrency=2](evidence/stage2_stream_c2.json), [concurrency=4](evidence/stage2_stream_c4.json).

### Source provenance

GitHub on Tang intermittently timed out on HTTPS; no source files were overwritten in the working tree. The SSE server and corrected sweep were copied from **GitHub** into ignored test snapshots and `git hash-object` was checked against the corresponding GitHub blobs. See the full [source manifest](evidence/stage2_source_manifest.json). Therefore `environment.git_commit` inside raw streaming JSON denotes Tang's older worktree HEAD, **not** the actual imported snapshot's source commit.

### Blockers still open

- **vLLM:** installed neither successfully nor tested. Official wheels (approximately 433 MB for version 0.19.1) failed due to network timeouts/throttling; this is NOT a vLLM performance comparison.
- **Nsight:** tools not installed/available; PyTorch profiler output is not GPU occupancy or DRAM bandwidth counters.
- **FSDP/multi-GPU:** only one NVIDIA GPU is available on Tang. No FSDP distributed-scaling claim.
- **Mac:** the remote device stopped responding during this phase; source updates used GitHub directly. This document does not claim Mac validation for these new commits.

## 3. Next acceptance gates

1. Improve or replace serialized request execution with an explicitly labeled scheduler, while retaining request/cost comparability.
2. Run vLLM with the identical request set after a compatible isolated installation succeeds; report TTFT/TPOT/token throughput/memory, not extrapolations.
3. Evaluate FSDP / two-rank DDP only where sufficient real GPU resources exist; preserve exact RNG and optimizer state across restart.
4. Collect real Nsight occupancy/memory data when tools are available before changing Triton dispatch policy.

## 4. Real PyTorch FSDP1 single-GPU execution (NO_SHARD)

Source commit: `d41a4c2`. On Tang's RTX 4090 Laptop, FSDP1 executed 20 forward/backward/AdamW steps on a synthetic 128→256→128 MLP with batch size 16 and FP32 parameters.

**Critical limitation:** PyTorch printed `FSDP is switching to use NO_SHARD` for world_size=1. This proves correct use of FSDP wrapping/training interfaces, **not actual parameter sharding, CPU offload, multi-GPU communication or FSDP scaling.**

| Metric | FSDP1 (NO_SHARD, 1 GPU) | DDP (1 GPU, same tensor dimensions) |
|---|---:|---:|
| Median step (ms) | 2.092 | 0.559 |
| Samples/s across 20 steps | 2148.4 | 2866.8 |
| Last loss | 0.22457 | 0.22457 |

FSDP-reported peak **Torch tensor allocated** GPU memory was **0.0181 GiB**, not the total GPU process memory seen by nvidia-smi. This small synthetic benchmark is not a representative LLM workload. Metrics include startup effects in each recorded run; single trial per method is not sufficient to claim one framework has general speed superiority.

Raw evidence: [FSDP](evidence/stage2_fsdp_single_gpu.json), [same-shape DDP](evidence/stage2_ddp_single_gpu_reference.json).

**Next gate:** genuine `FULL_SHARD` behavior and optimizer/checkpoint across at least two separate CUDA ranks. Tang currently has just one visible CUDA GPU, so that gate remains BLOCKED.
