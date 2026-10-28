# Stage 15 multi-track real GPU results — October 29, 2026

Source commit: `677b509`; all raw evidence in `evidence/stage15_multitrack` and `evidence/stage15_kernel_train`, with SHA256 manifests. Model experiment: Qwen3-8B-AWQ, vLLM 0.19.1, 16GiB Tang RTX 4090 Laptop, thinking disabled, previous K8s/prose frozen 24-request documents, new paired arrival seeds 8101 and 8201. 12 valid cases, 288 offered requests. **These are developmental, not independent holdout results.**

| Domain | Offered rate/s | Policy | Completed/24 | Rejected | All-offered TTFT<=250ms | Useful output tok/s |
|---|---:|---|---:|---:|---:|---:|
| k8s | 8 | FIFO400 | 17 | 7 | 58.3% | 270.3 |
| k8s | 8 | SJF400 | 18 | 6 | 66.7% | 266.9 |
| k8s | 8 | Full adaptive | 18 | 6 | 66.7% | 266.6 |
| k8s | 16 | FIFO400 | 17 | 7 | 29.2% | 335.3 |
| k8s | 16 | SJF400 | 18 | 6 | 33.3% | 296.4 |
| k8s | 16 | Full adaptive | 17 | 7 | 45.8% | 304.2 |
| prose | 8 | FIFO400 | 20 | 4 | 54.2% | 295.3 |
| prose | 8 | SJF400 | 20 | 4 | 54.2% | 295.4 |
| prose | 8 | Full adaptive | 17 | 7 | 66.7% | 308.4 |
| prose | 16 | FIFO400 | 14 | 10 | 4.2% | 296.1 |
| prose | 16 | SJF400 | 15 | 9 | 8.3% | 289.0 |
| prose | 16 | Full adaptive | 13 | 11 | 12.5% | 290.8 |

At higher offered rate, Full Adaptive sometimes has higher all-offered TTFT SLO coverage while rejecting at least as many offers and not consistently improving useful throughput. There is no demonstrated general superiority; the 250ms threshold may be inappropriate for these workloads. Recalibrate using request/service-time distributions and conduct independently sourced holdout comparisons with confidence intervals.

## GPU kernels and training (separate execution after serving shutdown)

Two CUDA shape cases: 128x1024 FP16 and 256x4096 FP16; correctness tested against PyTorch reference. Triton RMSNorm median speedups were 3.937x and 4.600x; Softmax 2.135x and 2.852x. These are measurements against the configured eager PyTorch baseline, not a claim against every fused or compiled implementation. Raw numeric results contain timing and error bounds.

FSDP1 12-step world_size=1 smoke passed, median ~1.576 ms/step. PyTorch selected `ShardingStrategy.NO_SHARD`. This is **not** proof of FSDP sharding, multi-GPU scaling, or memory savings. Initial wrapper failed because `torchrun` did not exist in the serving venv; the same script was successfully executed through the available `/home/you/.local/bin/torchrun` command, retaining the failure log. Future runner must select/validate the actual executable.

## Remaining experiment gates

1. Refreeze model-specific service-time predictors for 8B, eliminate 1.5B-based latency priors and run separate calibration/test sets.
2. Independently source longer, varied request sets, evaluate SLO curves (250/500/1000ms) and throughput with uncertainty intervals.
3. Add torch.compile and realistic fused-kernel baselines, more matrix shapes and reproducible profiler traces.
4. On a multi-GPU host only, verify actual FULL_SHARD and measure GPU memory and throughput; single GPU is an API smoke only.
