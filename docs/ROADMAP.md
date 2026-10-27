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
- [x] Real vLLM 0.19.1 service launched with Qwen2.5-1.5B on Tang RTX4090 Laptop; actual SSE evidence in Stage3.
- [x] Real SSE sweeps: 1/2/4/8 concurrence, two orders, 32-token cap; long output genuinely 128 tokens; Stage3 report.
- [x] All real serving requests had provider completion usage (100%); TPOT identified as client estimate.
- [x] Model snapshot, CUDA/PyTorch/vLLM versions, host/port/flags and workload seeds pinned; raw JSON persisted.

## M3 — Profiling
- [x] Real Nsight Systems CUDA kernel trace and stats CSV captured on Tang; see [Stage3 Nsight](STAGE3_NSYS_REPORT_2026-10-28.md).
- [ ] Nsight Compute per-SM occupancy/DRAM counters NOT collected; NVIDIA RmProfilingAdminOnly=1, no authorized privileged change.
- [ ] Link identified bottleneck to source line/kernel and quantify any fix.

## M4 — Distributed training
- [x] Single-process CPU smoke (3 steps) on Tang.
- [x] Single-rank exact DDP restart PASS on CPU and one CUDA GPU (model, optimizer and per-rank RNG states), [Stage 2 report](STAGE2_REPORT_2026-10-26.md).
- [ ] Two-GPU DDP scaling and restart still blocked: Tang exposes only one NVIDIA 4090 Laptop GPU.
- [x] Real FSDP1 single-GPU smoke: 20 steps, effective NO_SHARD, [Stage2 report](STAGE2_REPORT_2026-10-26.md). Not distributed sharding.
- [ ] A separate, properly sharded FSDP baseline (not inferred from DDP).
- [ ] Quantify token/s or samples/s; synthetic MLP benchmark is infrastructure smoke, not a trained LLM.

## Go/no-go
A milestone is PASS only with real logs, reproducible command, source commit and artifact. Missing GPU/driver/toolchain = BLOCKED, not PASS.

## 2026-10-26 additional work
- [x] Grouped RMSNorm config sweep: in-sample candidates and held-out shapes measured; no broad win, no default dispatch change.
- [x] Local-only serving runbook and warmup-enabled SSE request benchmark prepared.
- [x] vLLM v0.19.1 installed and real Qwen model server validated on Tang (Stage3 evidence).
- [x] Torch profiler diagnostic completed for 256×4096 FP32; source commit and sanitized operator table recorded. **Nsight remains pending**, no occupancy/DRAM claims.

- [x] vLLM wheel downloaded from verified mirror (official SHA256), installed and live served/benchmarked on 2026-10-28.
- [x] Separate offline Hugging Face GPU baseline: real Qwen2.5-1.5B inference measured at batch sizes 1,2,4; see RESULTS_2026-10-26.md. **Still not vLLM.**

## Stage 2 — 2026-10-26 verified
- [x] Real Transformers SSE server with corrected explicit `transformers_serialized` engine labeling, 24 streamed requests, valid token usage and reproducible summaries.
- [x] DDP Checkpoint V2 deterministic CPU + CUDA 3+5 vs 8-step recovery with exact optimizer/model/RNG state equality.
- [x] vLLM installation blocker resolved via verified wheel and real SSE testing (Stage3); Transformers baseline stays a distinct engine.
- [x] Mac local validation recovered and reran full CPU regression suite for Stage2/3/4.

## Stage 3 — 2026-10-28
- [x] vLLM 0.19.1 successful isolated install, runtime/model startup and real loopback request serving.
- [x] 192 short-answer requests over two orders at concurrency 1/2/4/8; all successful.
- [x] Corrected long decode actually generated 128 tokens, 36/36 requests; false earlier 21-token trial retained as negative evidence.
- [x] Raw JSON, configuration, numerical comparisons and output-length gate tests recorded in [Stage 3 report](STAGE3_REPORT_2026-10-28.md).
- [ ] Nsight Compute GPU hardware counter profiling, not yet accepted.

- [x] NVTX-targeted 30-call RMSNorm comparison: 150 eager launches vs 30 Inductor vs 30 Triton; archived [Stage3 NVTX report](STAGE3_NVTX_REPORT_2026-10-28.md).

## Stage 4 — Prefix caching (2026-10-28)
- [x] Real provider prompt_tokens captured from SSE; length/coverage gate.
- [x] vLLM prefix caching explicit enable and disable launch flags.
- [x] Frozen shared/distinct long-prefix experiment and Mac unit tests.
- [x] Corrected real cache ON/OFF × shared/distinct × concurrency 1/8 × two seeds on Tang (384/384 requests), 16/16 prefix-hit gates PASS. [Stage 4 report](STAGE4_PREFIX_CACHE_REPORT_2026-10-28.md).

- [x] First 384-request test identified and reproduced cross-case cache contamination in distinct-prefix control (order reversal exposed it); initial experiment **NOT scientifically accepted**.
- [x] Corrected distinct IDs and Prometheus gates; 384-request rerun PASS, see Stage4 report.

## Stage 5 — KV capacity, mixed-context, recurring plan audit

- [x] Frozen Stage5 experiment protocol and fail-closed bounded KV capacity eviction probe.
- [x] Mixed-context serving benchmark with actual 4-bucket prompt-token validation.
- [x] Half-hour GitHub Actions plan audit + CPU validation scheduled (best effort; cannot run Tang or write in ChatGPT).
- [x] Tang 256-block KV eviction verified by four cold-hit-evicted-rewarmed counter gates, see Stage5 capacity report.
- [x] Mixed-context real SSE 4 lengths x concurrency 1/2/4/8 x two seeds = 192 requests PASS, see Stage5 mixed report.
- [x] Stage5 raw 9 GPU JSON artifacts, SHA256 manifest and two analysis reports published to GitHub.

## Stage 6 — Ready next after Stage5 validation
- [ ] Freeze production-like heterogeneous prompt corpus with no repeated synthetic sentences, separate train/test-style workload IDs.
- [ ] Characterize KV cache collision/eviction under multiple block capacities (128/256/512) and optional prefix reuse; don't infer capacity from TTL alone.
- [ ] Build multi-seed latency-vs-throughput Pareto curve with explicit P95 TTFT SLO threshold, varied output lengths and fixed hardware/model.
- [ ] Evaluate reproducible engineering optimization with memory/performance logs, not just a repackaged demo.
