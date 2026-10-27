# Stage 3 — Targeted NVTX / Nsight Systems RMSNorm profiling

**Date:** 2026-10-28. **Source commit:** `1ae20a245e5d745f21cb5b976a6fecc76839a92c`.

## Why this follow-up matters

The earlier trace mixed benchmark framework fill/measurement kernels with RMSNorm and Softmax. The new experiment pins a single 256×4096 FP32 input, correctness-checks outputs and executes 30 measured calls per method inside dedicated NVTX ranges:

- `llm-opt:eager_unfused`: PyTorch unfused RMSNorm
- `llm-opt:torch_inductor_fused`: compiled PyTorch with output dtype cast fused
- `llm-opt:custom_triton`: custom one-row Triton RMSNorm

All variants passed torch.testing.assert_close with atol=rtol=2e-5.

## GPU operations attributed to NVTX ranges

| NVTX range | GPU kernel launches across 30 calls | Launches/call | NVTX projected GPU activity (ms)* |
|---|---:|---:|---:|
| `eager_unfused` | 150 | 5.0 | 0.860 |
| `torch_inductor_fused` | 30 | 1.0 | 0.752 |
| `custom_triton` | 30 | 1.0 | 0.432 |

*Instrumented GPU activity projected onto NVTX windows includes launch scheduling and synchronization; not a substitute for uninstrumented latency.

## Primary GPU kernels observed

| Method | GPU kernel executions | Mean kernel time (us) | Median kernel time (us) |
|---|---:|---:|---:|
| torch_inductor_fused | 30 | 4.563 | 4.544 |
| custom_triton | 30 | 4.552 | 4.544 |

**Interpretation:** eager PyTorch launches five GPU kernels for each RMSNorm, whereas both compiled PyTorch and custom Triton launch one. This confirms launch count / operator fusion as an important explanation for the improvement compared with eager operations.

Crucially, do not claim custom Triton universally beats compiled PyTorch. In a previous uninstrumented run at 256×4096 FP32, custom Triton was slower than Torch Inductor. The NVTX activity timing above includes different framework and instrumentation overhead and cannot refute the uninstrumented result.

## Archived source and derived evidence

- [Workload and numeric correctness](evidence/stage3_nsys_nvtx/workload.json)
- [NVTX GPU projection CSV](evidence/stage3_nsys_nvtx/nvtx_nvtx_gpu_proj_sum.csv)
- [NVTX kernel names and counts](evidence/stage3_nsys_nvtx/nvtx_nvtx_kern_sum.csv)
- [Full CUDA kernel statistics](evidence/stage3_nsys_nvtx/cuda_kernels_cuda_gpu_kern_sum.csv)
- [SHA256 provenance manifest](evidence/stage3_nsys_nvtx/manifest.json)

The raw binary trace stays in the Tang user-private ignored results path given by the manifest. Its SHA256 was recorded, but the binary was not uploaded to GitHub.

**Limitations:** GPU SM occupancy, DRAM bandwidth and warp stall counters are unavailable because NVIDIA RmProfilingAdminOnly=1. CPU sampling is restricted by perf_event_paranoid=4. No system configuration was changed to bypass permission controls.

## Next gate

Do multiple independent uninstrumented measurements under a stronger shape/workload matrix, and collect privileged Nsight Compute counters only if permitted by an authorized administrator.
