# Stage 4 — Actual vLLM Prefix Cache ON/OFF Factorial Ablation

**Date:** 2026-10-28. **Status:** PASS for this controlled synthetic workload. Numbers below are calculated from real Tang GPU inference, token-usage and vLLM Prometheus prefix-token counters.

## Runtime identity

- Device: Tang, NVIDIA RTX 4090 Laptop GPU, driver 580.178.04, vLLM 0.19.1, PyTorch 2.10.0+cu128, Triton 3.6.0.
- Model: pinned Qwen2.5-1.5B-Instruct FP16, Hugging Face snapshot 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- Server: local 127.0.0.1:8026 only; max-model-len 2048, max-num-seqs 8, GPU memory utilization 0.50.
- **Experiment source commit**: d3f34614f133f9d9b5f071944abb472510fc99f9. **Post-processing source commit**: ac4ea0fb1bb4d0d5df3834c4bdfadfb163e2d6ee.
- Cache arms created in **different server processes** using explicit --enable-prefix-caching versus --no-enable-prefix-caching switches, both confirmed in each actual engine initialization log.
- Both servers were shut down after their own cases; no global system software was modified and the GPU was returned to idle.

## Experimental design

- 2 server cache states × 2 request-prefix modes × 2 concurrency levels (1/8) × 2 seeds (42/137).
- 24 successful actual SSE requests per case, plus 2 excluded warmups per case: **384 measured requests** and 32 warmups.
- Real server-reported mean prompt length about **1,208–1,209 tokens** per request; each individual prompt was required to fall within [900, 1450].
- All 16 cases passed real input token coverage, output token coverage, all-request-success and vLLM prefix cache counter acceptance.
- Shared prefix: identical 35-repetition technical text preceding unique request suffix. Distinct negative control: first-content token identifier varies by seed, concurrency and request index to prevent reuse between cases.
- For seed 42: case order shared 1 → distinct 1 → shared 8 → distinct 8. For seed 137 the order is reversed, controlling a portion of order-related drift.

## Descriptive effect with two seeds

Cache ON/OFF comparisons are **paired by prefix mode, concurrency and seed**, with the same request generator/length. A ratio greater than one in TTFT OFF/ON means cached requests had lower TTFT. A ratio greater than one in throughput ON/OFF means caching improved aggregate throughput.

| Prefix mode | Concurrency | TTFT improvement (OFF / ON; mean of 2 seed ratios) | Throughput improvement (ON / OFF; mean) | Mean cache ON hit fraction (queried tokens) |
|---|---:|---:|---:|---:|
| shared | 1 | 4.11× (range 3.93–4.30×) | 1.37× (range 1.37–1.38×) | 96.7% |
| shared | 8 | 10.71× (range 10.00–11.43×) | 3.42× (range 3.41–3.44×) | 99.3% |
| distinct | 1 | 1.00× (range 1.00–1.00×) | 1.00× (range 0.99–1.00×) | 2.3% |
| distinct | 8 | 0.96× (range 0.90–1.02×) | 1.03× (range 1.00–1.05×) | 1.3% |

**Finding:** Prefix caching has a strong effect only when the long prefix is actually reused. In the same synthetic workload, cache ON without reusable prefix gives little/no TTFT advantage. The vLLM prefix-token hit fraction for shared-prefix traffic is around 94–99%, while the corrected distinct-prefix control remains approximately 1–3%. Cache OFF counters remain zero.

## Matched per-seed raw comparisons

| Mode | Seed | Concurrency | OFF P95 TTFT (ms) | ON P95 TTFT (ms) | OFF/ON TTFT | OFF throughput (tok/s) | ON throughput (tok/s) | ON hit % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared | 42 | 1 | 72.9 | 18.6 | 3.93× | 101.0 | 137.9 | 94.2% |
| shared | 137 | 1 | 74.3 | 17.3 | 4.30× | 100.2 | 138.6 | 99.3% |
| shared | 42 | 8 | 424.2 | 42.4 | 10.00× | 260.4 | 887.9 | 99.3% |
| shared | 137 | 8 | 465.7 | 40.7 | 11.43× | 259.5 | 891.4 | 99.3% |
| distinct | 42 | 1 | 73.7 | 73.9 | 1.00× | 100.6 | 100.0 | 2.1% |
| distinct | 137 | 1 | 75.0 | 74.7 | 1.00× | 100.0 | 99.8 | 2.5% |
| distinct | 42 | 8 | 422.8 | 469.4 | 0.90× | 307.1 | 322.3 | 1.3% |
| distinct | 137 | 8 | 469.2 | 460.7 | 1.02× | 260.0 | 261.3 | 1.3% |

All server cache state assertions and metrics were verified in programmatic acceptance gates. The metric token-hit deltas and raw per-request responses are preserved, not inferred from TTFT.

## Initial flawed run and root-cause correction

The initial experiment executed 384 real requests and passed HTTP/input-length checks, but had **invalid negative controls**: distinct-prefix IDs varied only by seed + request index, so identical prompts were replayed in a later concurrency case. With cache ON, whichever distinct-prefix case ran **second** reused ~99.3% of queried tokens. Reversing the seed's order exposed the contamination.

- Initial flawed source commit: 9beb86b9c878d1d0f1fba40372b85b0df6ff170c.
- Corrected source commit: d3f34614f133f9d9b5f071944abb472510fc99f9.
- Correction: incorporate concurrency in distinct-prefix request IDs and fail closed when caching ON + distinct prefix hits exceed 10%; caching ON + shared requires ≥70% hits; caching OFF requires zero query/hit counters.
- The invalid run is **not counted as valid negative-control evidence**. Its original unmodified full files stay on Tang in ignored results, and all 16 case summaries, cache-counter deltas, source hashes and failure classification are archived in GitHub.

## Evidence links

- [Complete computed analysis (16 cases and 8 matched pairs)](evidence/stage4_prefix_cache/stage4_prefix_cache_analysis.json)
- [Raw corrected 32 JSON artifacts: 16 client request logs + 16 cache-counter gates](evidence/stage4_prefix_cache/stage4_prefix_cache_raw)
- [Full source commit, actual cache flags, runtime configs, SHA256 provenance](evidence/stage4_prefix_cache/stage4_prefix_cache_server_manifest.json)
- [Invalid first-run audit and raw-case fingerprints](evidence/stage4_prefix_cache/stage4_invalid_first_run_summary.json)
- [Frozen protocol and corrected gate](PREFIX_CACHE_PROTOCOL.md)

## Limitations and next acceptance

1. This is deliberately synthetic, extremely repetitive context; it favors cache reuse and does **not** represent realistic production user traffic.
2. Two seeds and a fixed test machine do not support an inferential confidence interval or universal speedup claim. Recorded ranges are descriptive.
3. TTFT includes network, queuing and server prefill; TPOT is a client-derived estimate, not a GPU microarchitectural counter.
4. Cache OFF was always measured before cache ON (each in a fresh server); within each server request order reversed across seeds. Negative controls constrain, but do not fully eliminate, server/startup/thermal drift.
5. Comparing prefix-cache flags is an application-level cache ablation, not a test of a custom caching algorithm. Do not claim we implemented PagedAttention/KV caching internals.
6. Larger models, longer context distributions, time-to-first-token under mixed traffic and cache eviction/pollution remain future experiments.
7. Nsight Compute GPU occupancy and DRAM counters are still unavailable with driver RmProfilingAdminOnly=1. This result uses actual vLLM Prometheus counters, not Nsight hardware counters.

**Next milestone:** measure heterogeneous prompt length distributions and prefix collision/eviction under constrained KV capacity, then compare throughput-versus-P95 TTFT at 1/2/4/8 concurrency without changing the model or target memory policy.
