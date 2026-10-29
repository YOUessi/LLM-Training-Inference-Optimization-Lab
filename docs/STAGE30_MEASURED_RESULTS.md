# Stage30: real Qwen3-8B-AWQ long-window, three-seed mixed cold/hot prefix-cache results

**Actual measured**: 18 complete real-GPU cells, 96 unique offers each, **1,728 offered and completed**, no rejections or unexpected errors, 3 frozen arrival seeds (3011/3029/3041), rate=8 offers/s, fixed **11.875-second offered window** vs prior 2.875-second 24-offer window. Fresh server started for each ON/OFF × cold/hot mix × seed. PSF-licensed source docs, distinct request IDs, real output tokens, strict label/detailed length gate, prompt usage, sampled server queue and KV cache usage. Source commit `969ea758`; source manifest and per-request arrival offsets verified and raw JSON archived in `evidence/stage30_long_mixed/`.

## Per-mix caching ON vs OFF (paired over three seeds, n=3)

| Hot-prefix share | Matched seeds | Mean useful output TPS change | Seed-level bootstrap 95% CI | Mean all-offered 500ms SLO change | Mean P95 TTFT change | Positive TPS seeds |
|---|---:|---:|---:|---:|---:|---:|
| cold0 | 3 | -0.07% | -0.25% to +0.03% | -0.35 pp | +25 ms | 2/3 |
| hot50 | 3 | +10.62% | +10.49% to +10.72% | +1.04 pp | -2371 ms | 3/3 |
| hot100 | 3 | +24.04% | +23.22% to +24.61% | +6.60 pp | -4736 ms | 3/3 |

Hot0 uses no intentionally shared reusable instruction header; hot50 shares an identical long header in half of 96 offers (24 label +24 detailed), hot100 shares that header in all 96. Within every ON/OFF pair, exactly the same textual prompt, request identity, output mode, arrival schedule and frozen input SHA were verified. Independent server fresh start per condition and unrelated short model warmup prevent cross-cell prompt caching; however GPU itself is already initialized, so these are **fresh-prefix**, not cold GPU boot measurements.

**Important statistical scope:** Only **three independent arrival seeds**; a seed-level 5,000-draw bootstrap has extremely coarse support, and the apparent intervals are exploratory, NOT evidence of population significance. Individual requests within a seed must not be treated as independent assignment units. These are 11.875-second offered windows (4.13× previous Stage14/25 24-offer 8/s windows), not an hour-long live production soak test. The stable prefix is deliberately engineered and not a naturally sampled repetition rate. vLLM prefix hit counters were not captured, so actual cache hit percentage cannot be asserted. The model may have seen some public standard library docs during pretraining.

## Full reported metrics and negative results

Every cell includes all-offered 250/500/1000/2000ms TTFT fractions, arrival-relative P95/P99 TTFT, useful decoded tokens/s, client executor wait, server peak waiting/running gauges, KV cache peak fraction, realized short/long output lengths, no-error accounting and request-level records. The 18 rows and nine paired comparisons are machine-checked in `analysis/stage30_real_analysis.py` and `evidence/stage30_long_mixed/analysis.json`. Source/provenance and mixed-prefix composition are frozen in `docs/STAGE30_PROTOCOL.md` and `data/stage30_manifest.json`.

**Deployment gate:** This research experiment does not demonstrate production-traffic generalization, long-duration drift, or overall superiority at all loads. Do not enable a production rollout based on engineered hot-prefix throughput alone.
