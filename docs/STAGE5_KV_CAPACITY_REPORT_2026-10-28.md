# Stage 5 — Measured KV Cache Capacity and Eviction

**Date:** 2026-10-28. **Result: 4/4 acceptance gates PASS** on a real RTX 4090 Laptop GPU running vLLM 0.19.1 and Qwen2.5-1.5B-Instruct FP16.

## Reproducible capacity constraint

- Git source: `e7c96b8f9568885074630f4727e274a31cb862ed`.
- vLLM ran with prefix caching explicitly ON and num-gpu-blocks-override=256; context window 2048, max-num-seqs 8.
- The exact same 1,211-token anchor prompt was queried twice, followed by 12 distinct-prefix ~1,210-token requests, then the anchor queried twice again.
- Hit ratios were derived from vLLM Prometheus prefix-cache queried-token and hit-token deltas, not inferred from speed alone.

| Stage | Actual Prompt Tokens | Cache Hit Rate | TTFT (ms) |
|---|---:|---:|---:|
| Initial miss | 1211 | 0.0% | 92.6 |
| Immediate replay | 1211 | 99.1% | 17.5 |
| Replay after twelve new prefixes | 1211 | 1.3% | 68.6 |
| Rewarm replay | 1211 | 99.1% | 16.2 |

## Conclusion and interpretation

The sequence cold -> cached -> no longer cached under pressure -> cached again supports eviction or loss of retained prefix blocks under a constrained KV Cache. It does not expose every block internal state, prove any specific LRU implementation, or isolate effects of PagedAttention internals.

Limiting cache to 256 GPU blocks does not imply GPU total memory is reduced to the same size: model weights and CUDA graphs occupy memory independently. The experiment is synthetic and single-GPU.

## Evidence and limitations

- [Raw 16 request events and counter deltas](evidence/stage5_kv_and_mixed/stage5_eviction_20261028/capacity.json)
- [Source code, model snapshot and SHA256 evidence manifest](evidence/stage5_kv_and_mixed/stage5_manifest.json)
- [Frozen pre-run protocol](STAGE5_PROTOCOL.md)
- Server logs stay on Tang; only their SHA256 hashes were archived publicly.
- This is one capacity configuration and one request order; multiple capacities and seeds are needed to locate repeatable eviction thresholds.

## Next experiment

Repeat with alternate KV block budgets, mixed prefix-reuse patterns and realistic prompt distributions, report latency and cache hit count at each load.
