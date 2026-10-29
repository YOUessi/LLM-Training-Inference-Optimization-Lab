# Stage27 Qwen3-shaped CUDA fused kernel and PyTorch Inductor comparison

**Measured on the real Tang RTX 4090 Laptop GPU**, source pinned `9bf5b5f`, 8 correctness-checked cases against exactly matched PyTorch eager and `torch.compile`/Inductor, three repetitions per case; raw JSON, profile operator table, exact source and hashes in `evidence/stage27_measured`.

| Workload | Triton median ms | Inductor median ms | Triton speedup vs Inductor | Eager speedup |
|---|---:|---:|---:|---:|
| Fused residual-add + gamma RMSNorm, 64×4096 FP16 | 0.00819 | 0.00819 | 1.000× | 5.467× |
| Fused residual-add + gamma RMSNorm, 256×4096 FP16 | 0.01763 | 0.02016 | **1.143×** | 4.646× |
| Fused residual-add + gamma RMSNorm, 1024×4096 FP16 | 0.07066 | 0.07373 | 1.043× | 3.420× |
| Fused residual-add + gamma RMSNorm, 64×4096 BF16 | 0.00819 | 0.00819 | 1.000× | 5.475× |
| Fused residual-add + gamma RMSNorm, 256×4096 BF16 | 0.01830 | 0.02150 | **1.175×** | 4.476× |
| Fused residual-add + gamma RMSNorm, 1024×4096 BF16 | 0.07029 | 0.07398 | 1.053× | 3.443× |
| Masked causal *materialized* Softmax, 32 heads × seq256 | 0.02109 | 0.02310 | 1.096× | 2.917× |
| Masked causal *materialized* Softmax, 32 heads × seq512 | 0.07069 | 0.09491 | **1.343×** | 3.344× |

The fused RMSNorm uses Qwen3 actual 4096 hidden dimension, gamma weights and residual input with intermediate dtype rounding, not merely an unweighted toy kernel. Both candidates and compiled Torch passed relative/absolute correctness checks and max-error recording. Representative Torch profiler captures GPU operator-level self time, **not Nsight Compute occupancy/DRAM counters**. These are microbenchmarks with small timing differences: improvement claims are conditional on tested sizes and hardware, not end-to-end Qwen3 acceleration. Materialized masked Softmax is particularly **not a drop-in for FlashAttention**, the production fused attention used by Qwen3. Headline result is a modest 14–18% RMSNorm fusion speed advantage over an optimized compiler for 256 tokens on this GPU, not universal superiority.

An initial run failed due `triton.testing.do_bench(...quantiles=[.5])` returning a scalar float instead of a list; its original source/log are preserved in `evidence/stage27_invalid_v1` and excluded from the eight corrected cases. The benchmark implementation now handles scalar and list timing results and re-ran all 8 cases from pinned `9bf5b5f`.
