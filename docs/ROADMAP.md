# Milestones and acceptance gates

## M0 — skeleton, CPU smoke and baseline contracts
- [x] Independent source tree and dedicated Git history.
- [x] Standard-library CPU correctness oracles.
- [ ] CI is green on GitHub (cannot pass until remote repository exists).
- [x] Mac Python 3.12: 24/24 CPU unit tests green; Python compileall and bash -n green (2026-10-26).

## M1 — GPU kernel performance
- [x] CUDA/Triton available on Tang 4090 Laptop; environment and exact versions recorded.
- [x] RMSNorm and Softmax passed PyTorch reference FP16/FP32 on five shapes; also compared against properly fused torch.compile on three shapes (RMSNorm has confirmed regressions). See RESULTS_2026-10-26.md.
- [x] Config sweep (16 variants × 3 shapes) and independent validation (5 shapes, CUDA graph timing); mixed/negative results documented.
- [ ] Nsight profiling required to determine DRAM/occupancy bottlenecks before dispatch change.

## M2 — Serving on vLLM
- [ ] A real vLLM service is launched on NVIDIA GPU.
- [ ] Benchmark P50/P95 TTFT and P95 latency under fixed request lengths/concurrency.
- [ ] TPOT and token/s reported **only** when completion token usage is available.
- [ ] Keep server arguments, model hash and workload constant across variants.

## M3 — Profiling
- [ ] Capture Nsight Systems trace and Nsight Compute report.
- [ ] Link identified bottleneck to source line/kernel and quantify any fix.

## M4 — Distributed training
- [x] Single-process CPU smoke (3 steps) on Tang.
- [ ] Two-GPU DDP benchmark blocked: Tang currently exposes one 4090 Laptop GPU; single-GPU DDP + checkpoint creation passed, restart check pending.
- [ ] A separate, properly sharded FSDP baseline (not inferred from DDP).
- [ ] Quantify token/s or samples/s; synthetic MLP benchmark is infrastructure smoke, not a trained LLM.

## Go/no-go
A milestone is PASS only with real logs, reproducible command, source commit and artifact. Missing GPU/driver/toolchain = BLOCKED, not PASS.

## 2026-10-26 additional work
- [x] Grouped RMSNorm config sweep: in-sample candidates and held-out shapes measured; no broad win, no default dispatch change.
- [x] Local-only serving runbook and warmup-enabled SSE request benchmark prepared.
- [ ] vLLM runtime: dedicated Tang env targets v0.19.1 with preinstalled Torch 2.10.0 read-only; installation/server validation in progress; do not claim live service until actual requests pass.
- [x] Torch profiler diagnostic completed for 256×4096 FP32; source commit and sanitized operator table recorded. **Nsight remains pending**, no occupancy/DRAM claims.
