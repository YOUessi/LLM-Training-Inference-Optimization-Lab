# Stage 12 — Cross-Domain Held-Out Admission Control Validation

**2026-10-29. Genuine GPU experiment: 288 offered requests, 278 completed, 10 explicitly rejected, zero unexpected errors.** All original model-serving JSON and independent analysis were verified against SHA-256 hashes.

## Research question and controls

Stage 11 found the 150 ms bounded admission deadline more favorable than 400 ms on a multilingual Kubernetes documentation workload. Stage 12 changes the source domain to independently pinned classical English prose and natural-science passages, tests new arrival seeds 4093 and 5087, but holds model, request rate, input-token budget, client policy, output modes and all-offered SLO accounting fixed.

- Frozen 24 requests are drawn from five publicly attributed classic books and represent fantasy, natural science, science fiction, social fiction and travel fiction; original works were catalogued as US public domain by Project Gutenberg.
- Original actual Qwen2.5-1.5B tokenizer-calibrated prompt targets: 170, 300, 470 and 680 Tokens; six unique requests per bucket. Source dataset SHA256: fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04.
- Real vLLM 0.19.1, Qwen2.5-1.5B-Instruct FP16, one Tang RTX4090 Laptop GPU, max-num-seqs 8, prefix caching OFF, all model requests localhost-only.
- Two independently held-out seeds, conditioned fixed-window stochastic arrival schedules with matched mean offered rate 8 requests/s. This is NOT an unconditioned homogeneous Poisson process.
- A baseline unlimited admission and two client-side bounded policies, each with at most eight in-flight requests and four waiting slots, using 150 ms or 400 ms waiting deadlines.
- Two output modes (short label and detailed answer) × three policies × two seeds × 24 offered requests = 288 offers.
- Every explicitly rejected offer counts against timely first-Token compliance. Rejected traffic is never omitted from the SLO denominator; rejected/failed means all-offered P95 latency is not a finite observable statistic.
- Original experiment source: 385cc86535781f5dec92fbd5d59faa4e5bc47a36.

## Measured aggregate: 48 offered requests per output mode and policy

| Mode | Policy | Completed | Rejected | First Token within 250 ms, all offered | Minimum observed useful output Tokens/s |
|---|---|---:|---:|---:|---:|
| label | No admission | 48/48 | 0 | 100.0% (48/48) | 56.1 |
| label | Bounded 150 ms | 48/48 | 0 | 100.0% (48/48) | 56.1 |
| label | Bounded 400 ms | 48/48 | 0 | 100.0% (48/48) | 56.1 |
| detailed | No admission | 48/48 | 0 | 72.9% (35/48) | 747.8 |
| detailed | Bounded 150 ms | 41/48 | 7 | 85.4% (41/48) | 663.3 |
| detailed | Bounded 400 ms | 45/48 | 3 | 87.5% (42/48) | 682.1 |

## Critical comparison: the preferred threshold changes with workload

| Source content | Unlimited detailed | Bounded 150 ms detailed | Bounded 400 ms detailed |
|---|---:|---:|---:|
| Stage 11 Kubernetes (3 languages) | 72.9% (35/48), 0 rejects | 87.5% (42/48), 6 rejects | 81.2% (39/48), 4 rejects |
| Stage 12 literary/natural-science prose (English) | 72.9% (35/48), 0 rejects | 85.4% (41/48), 7 rejects | 87.5% (42/48), 3 rejects |

In Stage 11, 150 ms achieved 87.5% compared with 81.2% for 400 ms. Stage 12 reversed that ordering: 400 ms achieved 87.5% compared with 85.4% for 150 ms, while also rejecting fewer requests (3 instead of 7 of 48 detailed offers). The fixed static cutoff preferred by one workload is therefore **not reliably preferred by the other tested workload**.

This observation does not isolate domain as the sole cause: Stage 11 used Chinese/Japanese/English technical documentation while Stage 12 uses English-only classic prose; output token behavior and prompt distribution may also change. The differences are descriptive and from only two seeds, not statistical proof of adaptive optimization.

All label-mode controls served 48/48 offers and met the 250 ms first-Token target under every policy, with zero rejected requests.

## Original-genre refusal counts, detailed mode only

| Genre | Total offered per policy | 150 ms rejected | 400 ms rejected |
|---|---:|---:|---:|
| fantasy | 10 | 1 | 1 |
| natural science | 10 | 2 | 2 |
| science fiction | 10 | 3 | 0 |
| social fiction | 10 | 0 | 0 |
| travel fiction | 8 | 1 | 0 |

These genre buckets have only 8–10 offered requests per policy; rejection differences must not be characterized as stable fairness effects.

## Automation and audit trail

- Tang user-cron and GitHub Actions use a ten-minute cadence, but GitHub scheduled events are best-effort and local cron only runs while Tang is online.
- The prior Stage 11 jobs, the new seed-4093/5087 GPU jobs, and the Stage 12 CPU analysis all finished under reviewed immutable code and passed independent checks; the approved finite queue is now exhausted.
- A filesystem lock, immutable Git SHA, clean worktree check, GPU process and localhost port check prevent unauthorized or overlapping GPU experiments. The auto-executor does not automatically pull code, install packages or invent further jobs.
- Raw original service logs are retained on Tang with their SHA256 in the manifest; complete actual per-request model JSON and automation state are included in the GitHub evidence.

## Evidence

- [Twelve real Stage 12 SSE service records](evidence/stage12_crossdomain/).
- [Machine-generated exact two-seed cross-domain summary](evidence/stage12_crossdomain/stage12_analysis.json).
- [Full source versions, real runtime and per-file SHA256](evidence/stage12_crossdomain/stage12_manifest.json).
- [Finite approved automation completion status](evidence/stage12_crossdomain/stage12_auto_progress_state.json).
- [Preregistered Stage 12 experimental protocol](STAGE12_PROTOCOL.md).
- [Dataset source, author and US-public-domain caveats](../data/STAGE12_SOURCE_RIGHTS.md).
- [Earlier Stage 11 multilingual technical-doc holdout](STAGE11_REPORT_2026-10-28.md).

## Limitations and next research requirement

1. Only two new arrival seeds and 24 offers per cell. No confidence intervals or guaranteed production P95 SLO.
2. Stage 12 text differs in both topical domain and language composition, so the exact cause of the threshold preference change cannot be identified.
3. The tested admission controller is a client-side semaphore, not an internal vLLM scheduler; no adaptive, learned, or fairness-optimal algorithm is proven.
4. All-offered timeliness improves at the cost of explicit rejections and reduced useful throughput. No accepted-only P95 should replace these measures.
5. The source texts are classic public-domain prose, not modern production interactive conversations. Further workloads, more seeds, and multiple hardware/model sizes are required.

**Next:** Define a real workload-aware policy with a predeclared utility combining all-offered timely completion, explicit rejection cost and useful output throughput, then validate on a separate frozen dataset and independent arrival trials. Separately revisit the original Kernel/training optimization main tracks.
