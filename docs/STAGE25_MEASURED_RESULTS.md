# Stage25 Qwen3-8B-AWQ real serving capacity/prefix/KV/output experiment

**Measured 2026-10-29.** 22 real vLLM GPU cases × 24 distinct requests = **528 offered / 528 completed**, zero unexplained errors and zero explicit rejects. Corrected source SHA `9e19abf` is pinned in the immutable detached worktree; raw case JSON, source and SHA256 archived in `evidence/stage25_measured` with machine analysis at `analysis.json`. The original Stage25 `be540be` 13-case run was invalid under the long-answer realization gate and preserved in `evidence/stage25_invalid_v1` **excluded from these measured totals**. All results are from a local 16GiB RTX 4090 Laptop, Qwen3-8B-AWQ, vLLM 0.19.1, no thinking, one GPU, four PSF-derived content length buckets (~423/773/1173/1623 Qwen3 tokens), two arrival seeds and exact 8/16 fixed-window offers/s.

## Pairwise factor results

| Pre-frozen factor comparison | Matched pairs | Mean change in 250ms first-token SLO | Average useful output throughput change | Positive throughput pairs |
|---|---:|---:|---:|---:|
| **Shared-prefix cache ON vs OFF** | 8 | **+63.0 percentage points** | **+92.3%** | 8/8 |
| max-num-seqs 16 vs 8 | 8 | +8.85 percentage points | +13.7% | 8/8 |
| Reduced GPU memory utilization 0.82 → 0.70 | 2 | 0 points | −0.10% | 0/2 |
| Output cap 128 → 256 | 2 | 0 points | +50.2% | 2/2 |
| **Distinct-prefix negative control: cache ON vs OFF** | 1 | 0 points | +0.63% | 1/1 |

All-offered 250/500/1000ms TTFT, **per-case p95 measured TTFT**, server running/waiting peaks, GPU KV cache fraction, real output lengths, and per-request identifiers are in the raw evidence. Every comparison verifies identical frozen request IDs, text SHA and scheduled arrival offsets. The output-cap comparison is NOT a pure scheduling speedup: the output length and token demand change, and it must not be counted as a free throughput gain.

**Interpretation:** Under the deliberately engineered reusable 4-paragraph instruction prefix, prefix caching had a large measurable effect on both TTFT SLO and useful token throughput. The distinct-prefix control did NOT exhibit a comparable benefit, consistent with reuse being the relevant condition. The Qwen3 model was warmed up with a prompt from the same shared corpus before each measured case; this favors warm prefix reuse and is an explicit part of the experimental regime, not proof for cold-start traffic. The study does not measure prefix cache hit counters directly and cannot infer cache-hit percentage solely from throughput.

**Limitations:** 24 offers/cell and only two randomized arrival schedules, sequential server starts rather than fully randomized, and synthetic questions framing original PSF docstrings. Not sampled production traffic. No statistical significance or universal production-quality claims. These results establish *conditional* performance improvements on a real model/GPU. Capacity, KV and prefix factors have potential interactions; deployment requires longer held-out workload and latency distributions under target SLOs. Standalone client adaptive dropping still fails independent Stage23 original gates.

## Follow-up gates

- Test genuine traffic with recurring/unique prefix mixtures and cold cache, larger 100+ requests/cell, multiple seeds; report bootstrap CIs.
- Compare KV block budgets and actual cache hit metrics, not merely `gpu-memory-utilization` envelope.
- Validate ≥250/500/1000ms TTFT SLO after choosing a realistic service-level objective, and report useful throughput without hiding rejects.
