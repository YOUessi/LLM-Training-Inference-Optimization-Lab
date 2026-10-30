# Stage37: human OASST1 requests, 384 generated-token cap and actual vLLM prefix-hit counters

**Actually measured:** 12 cases of identical original, license-cleared filtered OASST1 English/Spanish/Chinese public community user requests, 96 offers/cell = **1,152 completed real Qwen3-8B-AWQ responses**, three frozen arrival seeds, 4 req/s, cold0 (no synthetic common system prompt) or hot50 (common 198-token system prefix in half the 96 requests), cache ON/OFF. 384 output-token budget compared with previous Stage35 128-token benchmark. Fixed Qwen3 model, vLLM fresh server, no-thinking, tokenizer prompt usage, server queue/KV gauges and prompt identity checks. Source text retained locally only; numeric results and file hashes in `evidence/stage37_long_answers/`.

| Original human prompt mix | Mean ON vs OFF useful TPS | Mean 500ms SLO change | Mean P95 TTFT change | Real vLLM cache hit token fraction ON | Responses reaching 384-token cap (ON) |
|---|---:|---:|---:|---:|---:|
| cold0 | -0.00% | +0.00 pp | -3.4 ms | 0.0% | 83.3% |
| hot50 | +6.78% | +1.74 pp | -1848.2 ms | 67.4% | 77.8% |

**Direct mechanism evidence:** `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total` are actual Prometheus counters of *queried/cached prefix Tokens*, sampled immediately after an unrelated warmup and again after the fixed 96-offer window; the difference is the actual token hit fraction for caching ON. It is **not** an inferred hit rate from throughput and is `null` for OFF when the engine did not query a cache. The fixed app system prefix is engineered and should not be called an observed naturally-occurring production cache distribution.

**Output truncation audit:** Six matched 4/s hot50 cases compare old cap128 and new cap384, with matching user prompts and arrival schedules but different generated token demand. On average **95.8%** of old responses hit 128 cap versus **77.8%** of new responses hitting 384 cap. Mean realized output length changed from **125.7 to 346.5** tokens. If a substantial fraction still hits 384, **do not** claim complete human-usable answers or a production response-quality SLA. Changes in latency/throughput across caps are due partly to decode work changes, not pure caching.

**Limits:** Three seeded 96-offer windows at 4/s, public community prompts filtered and artificially language balanced, engineered shared app prefix, no hour-long customer traffic; seed-resampled intervals have low resolution and do not cover model/traffic populations. The original human text is not committed. Prefix hit fraction measures KV prefix token queries, not global per-request prefix cache hit incidence or semantic correctness. Model quality and instruction following are separate Stage36 measurements.

Reproduction: `docs/STAGE37_PROTOCOL.md`, `scripts/run_stage37_long_answers.sh`, `analysis/stage37_outputcap_analysis.py`, `evidence/stage37_long_answers/` and previous `evidence/stage35_real_human/`.

**Analyst-code correction:** The first post-hoc Stage37 analysis invocation encountered an unbound local variable `d` in the cache-counter audit; no GPU run was invalid or re-executed. Corrected source references `x` (the current measured case), validated all 12 original immutable JSONs and matched source schedule. This is an analysis-only bug fix, not a retuned experiment.
