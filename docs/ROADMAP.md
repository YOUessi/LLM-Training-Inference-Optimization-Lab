# Milestones and acceptance gates

## M0 — skeleton, CPU smoke and baseline contracts
- [x] Independent source tree and dedicated Git history.
- [x] Standard-library CPU correctness oracles.
- [ ] CI is green on GitHub (cannot pass until remote repository exists).
- [x] Mac Python 3.12: 24/24 CPU unit tests green; Python compileall and bash -n green (2026-10-26).

## M1 — GPU kernel performance
- [ ] CUDA/Triton available and GPU environment recorded.
- [ ] RMSNorm and Softmax pass torch reference for FP16/FP32 at multiple widths.
- [ ] Track max absolute error, perf latency and pinned shapes; do not claim speedup without warmup and GPU synchronization.

## M2 — Serving on vLLM
- [ ] A real vLLM service is launched on NVIDIA GPU.
- [ ] Benchmark P50/P95 TTFT and P95 latency under fixed request lengths/concurrency.
- [ ] TPOT and token/s reported **only** when completion token usage is available.
- [ ] Keep server arguments, model hash and workload constant across variants.

## M3 — Profiling
- [ ] Capture Nsight Systems trace and Nsight Compute report.
- [ ] Link identified bottleneck to source line/kernel and quantify any fix.

## M4 — Distributed training
- [ ] Single process CPU smoke.
- [ ] Two-GPU DDP benchmark and checkpoint/restart verification.
- [ ] A separate, properly sharded FSDP baseline (not inferred from DDP).
- [ ] Quantify token/s or samples/s; synthetic MLP benchmark is infrastructure smoke, not a trained LLM.

## Go/no-go
A milestone is PASS only with real logs, reproducible command, source commit and artifact. Missing GPU/driver/toolchain = BLOCKED, not PASS.
