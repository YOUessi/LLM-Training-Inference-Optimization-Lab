# Stage 6 — Real GPU KV Cache Capacity Retention Matrix

**Date:** 2026-10-28. **Result: four separate vLLM model-process experiments completed, all validated as real observations.**

## Controlled design

- NVIDIA RTX 4090 Laptop GPU, vLLM 0.19.1, local Qwen2.5-1.5B-Instruct FP16.
- Real server caches initialized from scratch for each separate (KV Block capacity, filler count) case.
- All cases use same 35-repetition long technical prompt, seed 830, model context 2048, and single concurrent requester.
- Fresh server flags --enable-prefix-caching and --num-gpu-blocks-override were verified from each actual server log, with SHA256 recorded in provenance manifest.
- Cold miss, immediate identical replay, distinct prefix fillers, post-pressure replay and immediate second replay are recorded using real vLLM Prometheus queried-/hit-token deltas and SSE TTFT.

| Cache GPU blocks | Number of new distinct long prefixes | Anchor hit rate after pressure | Anchor TTFT after pressure | Classification |
|---:|---:|---:|---:|---|
| 128 | 4 | 1.3% | 68.3 ms | evicted |
| 256 | 4 | 1.3% | 67.1 ms | evicted |
| 512 | 4 | 99.1% | 15.8 ms | retained |
| 512 | 12 | 1.3% | 66.9 ms | evicted |

## Empirical conclusion

With the same four distinct ~1,210-token fills, the old long prefix loses its cache hit under 128- and 256-Block budgets but remains cached with 512 Blocks. When the 512-Block instance receives twelve such distinct prefixes, the old prefix also loses its hit.

**This is a block-budget/workload-retention relationship demonstrated by actual cache-hit counters.** It is not direct inspection of the internal replacement queue and does not establish an exact eviction threshold between these sampled points. Results may change with different context lengths, prompt entropy, concurrent clients or KV block-size details.

A retained anchor is an accepted measured outcome, not a failed test. Each fresh server still passed cold/warm/rewarm validation gates, and cache classification used the declared <=25% evicted, >=75% retained, otherwise partial rule.

## Provenance and raw evidence

- [Stage6 four-point machine summary](evidence/stage6_capacity_matrix/stage6_capacity_summary.json)
- [Four original SSE/Prometheus case traces](evidence/stage6_capacity_matrix/)
- [SHA256 of original files, actual launch flags and log fingerprints](evidence/stage6_capacity_matrix/stage6_capacity_manifest.json)
- [Frozen protocol](STAGE6_CAPACITY_MATRIX_PROTOCOL.md)

Raw server logs remain on Tang; the submitted evidence includes hashes only. No system-level driver permissions, GPU profiling restrictions, or unrelated projects were modified.

## Remaining Stage6 experiments

1. Expand to more block budgets and filler counts in independent sessions before fitting a retention threshold curve.
2. Replace repetitive text with a diverse, frozen corpus; study mixed shared/unique prefixes and cache pollution under realistic arrival processes.
3. Combine capacity sweep and SLO frontier (P95 TTFT constraints) to propose workload-dependent scheduling/cache budgets, with honest held-out validation.
