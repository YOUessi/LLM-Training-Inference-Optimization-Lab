# Stage27 — Qwen3-shape CUDA profiling and code-level optimization

Frozen benchmark compares a **fused residual-add + gamma-weighted RMSNorm** using Qwen3-8B's actual hidden=4096, eps=1e-6 and realistic token-row counts 64/256/1024 (FP16 and BF16), against correctness-checked PyTorch eager and TorchInductor `torch.compile` with identical outputs and intermediary dtype semantics. A separate causal masked softmax on heads=32, sequence lengths 256/512 is a **diagnostic** only. Qwen3 FlashAttention avoids materialized attention scores; standalone softmax speedup is not an end-to-end inference benefit. Each candidate runs 3 timing repetitions, correctness checks, max error, and torch.profiler CUDA operator self time on representative cases. No Nvidia hardware occupancy/DRAM claims unless verified by Nsight Compute counters.

The optimization contribution must be attributed to measured fusion, memory-traffic reduction, and correctness, not simply beating PyTorch eager while losing to torch.compile. Publish negative results openly. Run GPU only with a pinned source snapshot and the lab-wide lock; no concurrent serving/training.

## Amendment after invalid first execution

The original frozen Stage27 run halted at the first Qwen3-shaped RMSNorm benchmark after correctness checks: Triton v3.6 returns a scalar float for `do_bench(quantiles=[.5])`, but the driver incorrectly indexed `[0]`. This was a benchmark API bug, not a failed operator correctness result; original source SHA and stack trace are archived in `evidence/stage27_invalid_v1`. The revised driver accepts a scalar or list/tuple, and reruns **all eight cells** under a new frozen SHA. Previous failed results are never counted as measured.
