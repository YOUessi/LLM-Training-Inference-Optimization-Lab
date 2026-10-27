# Stage 3 — Nsight Systems real GPU kernel trace (2026-10-28)

Collected via NVIDIA Nsight Systems CLI 2026.5.1 (downloaded as a signed-publisher-repository .deb and SHA-256 verified, unpacked in an isolated Tang user directory without system installation).

## Environment and acceptance

- Profiled source commit: `2749782737d3873e103314814a4f62cf1cd19248`. Kernel workload: RMSNorm and Softmax, 256x4096 FP16, real CUDA RTX 4090 Laptop GPU (SM 8.9), driver 580.178.04.
- Captured actual CUDA API/kernel timeline with Nsight Systems using GPU tracing and CPU sampling disabled.
- The binary .nsys-rep is stored on Tang at `results/nsys_trials_20261028/rmsnorm_softmax.nsys-rep`; SHA-256: `9daa4ed511b4f57047434c70803b2534bc33a09eeefef772d420b02ce12995b1`. It was not uploaded to GitHub and is not claimed as a public attachment.
- GitHub evidence: [CUDA kernel statistics](evidence/stage3_nsys/kernel_stats_cuda_gpu_kern_sum.csv), [actual GPU benchmark JSON](evidence/stage3_nsys/kernel.json), [capture manifest](evidence/stage3_nsys/manifest.json), stdout/stderr.

## Captured GPU launch records

| Kernel name substring | Instances | Average instrumented GPU runtime (us) | Median instrumented GPU runtime (us) |
|---|---:|---:|---:|
| `_rmsnorm_kernel` | 226 | 7.21 | 9.09 |
| `_softmax_kernel` | 228 | 8.19 | 10.05 |
| `cunn_SoftMaxForwardReg` | 237 | 11.80 | 12.22 |

### Interpretation and methodological caveat

The GPU trace confirms that native Torch softmax and custom Triton RMSNorm/Softmax kernels genuinely launched on the device. Kernel microbenchmark tooling also generated many auxiliary filling/conversion kernels; one fill-related native kernel occupied most of the summed GPU time in this trace.

These **instrumented averages are not a substitute for the separate uninstrumented PyTorch versus Triton latency baseline**. Nsight instrumentation, baseline allocation/copy behavior, launch order and additional kernels can significantly alter per-invocation statistics. Do not sum nested profiler events to claim a global model speedup.

## Remaining limitation: GPU hardware performance counters

The NVIDIA driver exposes RmProfilingAdminOnly=1; the user does not have passwordless sudo. Linux perf_event_paranoid=4 blocks Nsight's CPU sampling. Nsight Systems CUDA trace succeeded with CPU sampling disabled, but hardware metrics such as achieved SM occupancy, DRAM bandwidth and Warp Stall breakdown from Nsight Compute have **not** been collected.

No driver/kernel configuration was changed to bypass profiler permissions.

## Next steps

Use explicit NVTX ranges around fixed-count, correctness-checked RMSNorm workloads to separate tool-harness overhead from kernel execution; if an authorized administrator enables profiler counters, independently capture Nsight Compute occupancy/DRAM/warp stall metrics, without overwriting the original negative results.
