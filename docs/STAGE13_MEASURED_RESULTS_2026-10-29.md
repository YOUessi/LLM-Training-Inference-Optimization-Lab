# Stage 13 — Real GPU results (raw archive transfer pending)

**Measured on 2026-10-29.** 32 experimental cells, 768 offers, 755 completed, 13 explicitly rejected, zero unexplained errors. All 32 corrected source-version request files and the strict 768-offer analysis were validated on Tang; the complete 44-file evidence archive has NOT yet been uploaded to GitHub. Treat this document as a measured preliminary report, not an online raw-evidence bundle.

## Novel mechanism and frozen comparison

The new client-proxy scheduler predicts each request's service time using input Tokens and output task type, updates a per-mode EWMA from completed requests, chooses an age-compensated shortest-predicted-job next, and dynamically adjusts admission waiting deadlines using active requests and queue depth. It is **not** a vLLM internal scheduler patch or a trained optimal algorithm.

Source code and experiment parameters were frozen before measured GPU execution in Git commit 4f3d4a5eadcbc41aef902ef1ee3da71b8a4c2683. The tested model is Qwen2.5-1.5B-Instruct FP16 on a Tang NVIDIA RTX 4090 Laptop GPU, served by vLLM 0.19.1 with prefix caching OFF. Each of the eight domain/seed/rate combinations ran unlimited, fixed150, fixed400 and adaptive policies on the exact same 24 offered documents, scheduled arrival times and alternating 12 short-label / 12 detailed output requests.

Domains: Kubernetes English/Chinese/Japanese technical documentation and English classic literary/natural-science prose (both previously used content domains, new arrival seeds). Arrival rates: matched randomized fixed-window 8 and 16 offers/second, **not an unconditioned Poisson process**. Seeds: Kubernetes 6101/6109; prose 7201/7209.

Primary utility was preregistered as U = all-offered 250ms first-token SLO compliance − 0.50 × rejection fraction + 0.20 × (useful completed output Tokens/sec divided by paired unlimited baseline throughput). A candidate win requires U exceed unlimited by at least 0.02, useful throughput at least 80% of unlimited, and rejects no more than 20% of offered requests. Rejected offers always count as SLO failures.

## Overall real-request accounting

| Policy | Offered | Completed | Explicitly rejected |
|---|---:|---:|---:|
| Unlimited | 192 | 192 | 0 |
| Fixed 150 ms | 192 | 184 | 8 |
| Fixed 400 ms | 192 | 191 | 1 |
| New adaptive | 192 | 188 | 4 |
| **Total** | **768** | **755** | **13** |

## All-offered 250ms first-token compliance at 16 requests/sec

| Domain | Seed | Unlimited | Fixed150 | Fixed400 | Adaptive | Adaptive utility − unlimited |
|---|---:|---:|---:|---:|---:|---:|
| Kubernetes | 6101 | 100.0% | 95.8% | 100.0% | 95.8% | −0.0612 |
| Kubernetes | 6109 | 83.3% | 87.5% | 79.2% | 87.5% | −0.0023 |
| English prose | 7201 | 100.0% | 95.8% | 100.0% | 100.0% | −0.00005 |
| English prose | 7209 | 75.0% | 87.5% | 75.0% | **91.7%** | **+0.1377** |

At 8 requests/second, all four strategies reached 100% 250ms first-token compliance across the four domain/seed pairs, and no general adaptive benefit was detected.

**Preregistered candidate-win counts:** Kubernetes 8/s: 0/2; Kubernetes 16/s: 0/2; prose 8/s: 0/2; prose 16/s: 1/2. The adaptive policy met the prespecified full utility/throughput/rejection threshold in only **one of eight matched settings**. Its result is therefore **negative overall with one local positive case**; do not claim general superiority.

In the prose 7209 / 16/s case, the adaptive arm raised all-offered timely-first-token coverage from 18/24 (75.0%) to 22/24 (91.7%). It completed 23/24 instead of 24/24, with output throughput falling from approximately 600.7 to 576.2 tokens/sec; this is a specific latency/rejection/throughput tradeoff, not a free performance gain.

## Experiment validation and two failures preserved

An initial source version (Git commit 76390668c59928d9528f74f6925e4216660a816f) generated 192 real offers but failed evidence validation because the result did not persist elapsed wall time, and cron's module import path was not configured. Those eight original raw case JSONs remain unmodified and are **excluded** from the corrected 768.

The corrected run source 4f3d4a5eadcbc41aef902ef1ee3da71b8a4c2683 includes elapsed_s, a pinned local source path and automated regression tests. Its first 192-offer seed was executed automatically. A CPU test then improperly assumed no prior GPU result files; this blocked further cron executions. The remaining three seeds (576 offers) were run **manually under the identical corrected Git SHA and the unchanged approved GPU script**, and all were separately verified by the original fail-closed data validator. This test isolation problem is documented; neither error is hidden or counted as an inference-model failure.

The 768-offer analysis recomputed per-case input/output token usage, completion/rejection accounting, all-offered 250ms SLO, and paired source/schedule identities from the original saved JSON. The true GPU service was stopped afterward. The entire Tang archive includes 32 corrected real-result JSONs, eight excluded original raw cases, strict machine analysis, provenance/issue state, and manifests (44 files). Its SHA-256 is 2143093893030c03f8f737e951d8a580ae87296d3de52f96eb37ed2f7005cb11.

**Archival limitation:** original 44-file archive and complete analysis are still preserved on Tang under results/stage13_export_full; a remote bulk transfer to Mac/GitHub was interrupted by a transfer-tool restriction. This report records results supported by the real locally verified source, but is **not a substitute for the missing online raw archive**. Do not claim that its 32 raw case logs were uploaded.

## Next gates

1. Complete authenticated raw-evidence transfer to GitHub, checking every archived SHA and preserving the original invalid version separately.
2. Repair and rerun the environment-independent cron test so it passes on both machines before authorizing any new automatic GPU job.
3. Ablate dynamic deadline adaptation and shortest-job aging independently on new frozen arrival seeds and documents, then evaluate longer traces with all-offered compliance, useful throughput, per-mode rejection and confidence intervals.
4. Resume the original Kernel and distributed-training optimization tracks rather than treating scheduling experiments alone as end-to-end training/inference optimization.

References: Stage13 preregistered experiment protocol (docs/STAGE13_PROTOCOL.md); workload/algorithm code (src/llm_opt_lab/serving/stage13_policy.py, stage13_mixed_bench.py, stage13_analysis.py); previous Stage11 and Stage12 measured reports.
