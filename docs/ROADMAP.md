# Milestones and acceptance gates

## Latest verified batch (2026-10-29): Stage25–28

- [x] Fail-closed **retire** old Stage13 cron with stale SHA; keep original logs/crontab. New finite batch scripts pin exact source SHA and require shared GPU flock, idle GPU, unoccupied localhost port, and no preexisting outputs.
- [x] Stage25: frozen licensed-source context/prefix/KV/output-length matrix, 22 real GPU cells, 528 completed real Qwen3-8B-AWQ requests, full all-offered TTFT/throughput/p95/queue evidence. Original invalid detailed-output length cohort archived separately.
- [x] Stage26: verified official Dolly license/source SHA and 512/96/96 frozen split; 5 real 8B QLoRA configurations, one validation-selected rank16 adapter, single-use final 96-example test. First attempt microbatch2 OOM archived; amended C auxiliary excluded from fair ranking.
- [x] Stage27: fused residual+weighted RMSNorm and causal materialized Softmax across actual Qwen3 model dimensions, compared to eager and torch.compile over 8 real CUDA cases; raw profiler/correctness evidence.
- [x] Stage28: full adapter+optimizer+data/CPU/CUDA RNG state checkpoint and exact step6 restart on a real pretrained Transformer adapter; four measured paths, hashes, negative speed-tradeoff evidence.
- [ ] Production-like serving traffic/cold-prefix behavior, more arrival seeds, confidence intervals, model end-to-end kernel replacement and deployment SLO.
- [ ] More real-task SFT data exposure and task-level human or robust semantic evaluation, outside the 96 fixed Dolly test.
- [ ] True multi-GPU DDP/FSDP FULL_SHARD and distributed checkpoint restart **blocked** on single 16GiB RTX 4090 Laptop.

**Research evidence validated; not production approval.** [Batch audit](../analysis/stage25_28_batch_acceptance.py), [Stage25](STAGE25_MEASURED_RESULTS.md), [Stage26](STAGE26_MEASURED_RESULTS.md), [Stage27](STAGE27_MEASURED_RESULTS.md), [Stage28](STAGE28_MEASURED_RESULTS.md).


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
- [x] Derived SLO frontier from Stage5 real mixed requests (two-seed worst P95 TTFT, min throughput): concurrency 1/2/4/8 qualify at 100/150/250/500ms SLO, respectively. [Report](STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md).
- [ ] Repeat SLO frontier with varied output lengths and nonrepetitive context to improve external validity.
- [ ] Evaluate reproducible engineering optimization with memory/performance logs, not just a repackaged demo.

- [x] Stage6 KV capacity matrix protocol frozen: 128/256/512 Block budgets, 4/12 fillers, fresh server per condition, retain/evict/partial classification.
- [x] Stage6 independent real-GPU capacity matrix: 128/256/512 blocks, 4/12 distinct long prefixes, 4/4 measured classification valid. [Report](STAGE6_CAPACITY_MATRIX_REPORT_2026-10-28.md).

## Stage 7 — Multi-genre nonrepeating prose, varied generation lengths
- [x] Freeze 30 attributable full-sentence passages from 5 classic public-domain-in-US works, exact original source SHA256 and excerpt SHA256.
- [x] Tang actual Qwen tokenizer calibrated 4 length buckets and preregistered prompt_tokens gates.
- [x] Tested Stage7 SSE workload: 24 unique docs/case × four context lengths × short/long decode × concurrency 1/2/4/8 × two seeds.
- [x] Real Tang Stage7 384/384 SSE requests accepted across 5 genres, 4 input buckets, two decode budgets, concurrency 1/2/4/8 and two seeds; original protocol-failed 120-request attempt retained. [Report](STAGE7_REPORT_2026-10-28.md).
- [x] Stage7 exploratory TTFT-vs-throughput frontier computed from two real seeds, all 16 raw case files + SHA256 manifest published, with explicit Stage5 distribution shift caveat.

## Stage 8 — Heterogeneous practical traffic and model output stopping

- [ ] Freeze genuinely diverse current-domain and multilingual text with explicit redistribution or permissive licenses; do not relabel old-book prose as enterprise production traffic.
- [ ] Build a workload with actual naturally early-stopping and long-running outputs, verified output token distributions rather than just different max_new_tokens values.
- [x] Implement and validate open-loop deterministic arrival scheduling with Poisson and matched-fixed-window variants; 288 real SSE requests per variant.
- [x] Measure client executor queue, arrival-to-first-token vs dispatch-to-first-token, rate-conditioned TTFT SLO violations and real vLLM KV cache gauges under fixed model.
- [ ] Publish Stage8 held-out validation and any measured model-serving improvements with failure/negative evidence.

## Stage8 — code/data frozen before live experiments
- [x] Pinned 24 real Kubernetes CC BY 4.0 excerpts across English, zh-cn, ja; original upstream commit+blob SHA and source attribution; not claimed cross-industry.
- [x] Frozen 24 Qwen-token-calibrated multilingual prompts at ~170/300/470/680 actual tokens; shared across mode and load comparisons.
- [x] Deterministic open-loop rate-based scheduler (2/8 req/s), label/summary/detailed output instructions, real token gate, client queue and vLLM KV gauge instrumentation.
- [x] 288 unconditioned Poisson real requests + separate 288 matched fixed-window real requests; all success, actual prompt/completion Tokens and mode gates PASS.
- [x] Stage8 analysis and source manifest archived: six matched mode/rate cells; unconditioned unequal-rate confound preserved as exploratory, not merged.

- [x] Stage8 unconditioned-Poisson 288 real requests measured, but empirical offered rates 1.48 vs 2.86 and 5.93 vs 11.45 cause unmatched-load confound; preliminary evidence retained.
- [x] Stage8 revised fixed-window randomized gaps designed; two seeded cases have identical actual average offered rate (not a Poisson process).
- [x] Stage8 fixed-window full 288-request matched-rate experiment verified with SHA256, 12 raw case records and full recomputed analysis; see Stage8 report.

## Stage 9 — Serving overload and admission-policy research

- [x] Stage9 load-generator A/B designed and CPU-tested: 8 vs 32 worker threads, server max-num-seqs fixed at 8, real server waiting/running Prometheus gauges and client queue accounting. [Protocol](STAGE9_PROTOCOL.md).
- [ ] Create a deadline-aware bounded queue/admission policy with explicit dropped/rejected/accepted success rates; never improve TTFT by silently discarding requests.
- [ ] Compare baseline vs controller under identical frozen multilingual requests and matched arrival schedules on the same GPU/model settings; use independent seeds and negative results.
- [ ] Report served useful output Tokens/s, acceptance %, error/timeout %, SLO violations over ALL offered requests, not only accepted ones.
- [ ] Expand to independent technical and nontechnical domains before claiming general policy effectiveness.


### Stage9 real-experiment gates
- [x] Stage9 8-vs-32 worker A/B: label@8 and detailed@2/8 req/s × 2 seeds = 288 real SSE requests PASS, 100% provider usage; unchanged source corpus/schedule.
- [x] Publish 12 raw real GPU cases, sampled server waiting/running gauges, source SHA256 manifest and queue-migration analysis. [Report](STAGE9_REPORT_2026-10-28.md).

## Stage 10 — Honest queue admission and SLO-aware load control

- [x] Implement and CPU-test client-side bounded admission (8 in-flight + 4 waiting slots + 250ms deadline) and explicit all-offered acceptance/rejection/error accounting. [Protocol](STAGE10_PROTOCOL.md).
- [ ] Run baseline vs admission under the same frozen multilingual input and fixed-window arrival schedules; compare ALL-offered-request P95, success coverage and useful output throughput.
- [ ] Add SLO and fairness analyses per language/length/output mode and a held-out seed or distinct data source; report negative outcomes.
- [ ] Only promote a scheduling technique after the all-offered denominator and independent validation show a genuine benefit.

### Stage10 acceptance gates

- [x] Frozen Stage10 A/B workload under the same server/input/arrival schedules and 32-worker request injector: unlimited vs bounded, label/detailed, 2 seeds.
- [x] Stage10 two-policy matched trial: 192 real offered requests, 186 completed, 6 explicitly rejected; all raw results and actual model usage archived. [Report](STAGE10_REPORT_2026-10-28.md).
- [x] All-offered SLO denominator verified; bounded detailed mode improved timely first-token fraction but reduced useful throughput and rejected 12.5% of detailed offered traffic. Full negative/positive tradeoff published.

## Stage 11 — Admission policy independent verification

- [ ] Freeze held-out arrival seeds and multiple offered rates including overload and low-load controls. Do not tune deadlines on the held-out schedules.
- [ ] Compare multiple queue slot/deadline configurations and explicit no-drop baseline using all-offered compliance, useful throughput, reject reasons and per-language fairness.
- [ ] Collect enough repetitions/requests to estimate uncertainty in tail SLO, and validate on at least one distinct domain before describing any production applicability.
- [ ] Keep new model/engine/KV cache optimizations separate from client admission-policy changes to avoid causal confounding.

## Stage 11 — 10-minute approved finite executor and held-out validation (2026-10-28)

- [x] Configure GitHub Actions schedule at 10-minute intervals (UTC 07,17,27,37,47,57); run regression, frozen Stage11 CPU preflight, and plan audit.
- [x] Implement local Tang user-cron tick with immutable approved Git SHA, single-instance flock, GPU/port preflight, bounded runtime and fail-closed state.
- [x] Freeze held-out seeds 2027 and 3031 and admission deadline A/B (unlimited, 150 ms, 400 ms) with original 24 frozen multilingual requests and full all-offered accounting.
- [x] Held-out seeds 2027 and 3031 completed all 288 actual offered requests (278 model completions, 10 explicit rejections, zero unexpected errors); original per-request SSE and server queue metrics preserved.
- [x] Independent all-offered analysis of 12 held-out cases completed; raw JSON, automation state, SHA256 manifest and Stage11 report archived in GitHub.
- [ ] Evaluate on non-Kubernetes domain/longer test windows before claiming generalization.

## Stage 12 — Beyond static client admission

- [ ] Create an independent rights-cleared workload outside Kubernetes technical docs; freeze before running GPU tests.
- [ ] Design workload-aware admission/queue priority using real input-length and output-length estimates, compared with static deadline admission while accounting for all offered and rejected requests.
- [ ] Use distinct train/tuning versus held-out loads, multiple offered rates, and repeated runs with independent seeds to verify throughput, SLO coverage, language fairness and actual rejection cost.
- [ ] Report negative results and compare against the simplest static baseline; never promote a method solely because accepted-only P95 improves.

## Stage 12 — New domain-shift held-out experiment (2026-10-28)

- [x] Freeze 24 source-attributed non-Kubernetes public-domain prose requests, five genres, four real Qwen-token-calibrated input lengths; exact SHA256 pinned.
- [x] Preregister 4093/5087 independent arrival seeds and no-admission/150ms/400ms client proxy policy comparison with 288 offered request accounting.
- [x] Extend 10-minute GitHub CPU preflight and finite Tang user-cron allowlist to Stage12 (two GPU seed jobs + CPU analysis); strict commit approval/GPU lock retained.
- [x] Stage12 two new seeds 4093/5087, 12 real GPU cases: 288 offered, 278 completed, 10 explicitly rejected, zero unexpected errors; immutable non-Kubernetes SHA and request/service metrics verified.
- [x] Published Stage12 independent-domain raw 12 cases, 15 evidence files, SHA256 manifest and cross-domain comparison: threshold ordering reversed (400ms beat 150ms under literary/natural-science workload). [Stage12 report](STAGE12_REPORT_2026-10-29.md).

## Stage 13 — Workload-aware, age-fair dynamic admission algorithm

- [x] Develop a real client-side adaptive scheduler combining Stage10 measured demand priors, request-specific input Tokens, per-mode EWMA, live active/waiting load, waiting-time aging and dynamic 100–400ms admission deadlines.
- [x] Freeze 768 real offers: 2 previously used source domains × 2 new seeds/domain × 2 fixed offered rates (8/16 req/s) × 4 policies × 24 mixed 12-label/12-detailed requests.
- [x] Define a predeclared all-offered optimization utility: SLO250 - 0.50×rejection fraction + 0.20×paired-baseline normalized useful TPS. Fail-closed correctness and policy preflight tests added.
- [x] Extend the 10-minute GitHub CI CPU check and finite source-pinned Tang executor with four safe Stage13 GPU seed jobs and independent 768-offer CPU analysis; no new arbitrary code execution.
- [ ] Run all 32 real Stage13 vLLM cells and verify original per-request SSE, output Tokens, model/queue gauges and no unseen model errors; preserve negative results.
- [ ] Publish 768-offer matched-domain A/B and compare adaptive vs unlimited and fixed150/fixed400 using preregistered utility, rejection/throughput/SLO and per-task fairness.
- [ ] Ablate priority versus deadline adaptation on new independent input sources before claiming a general-purpose model-serving improvement.
