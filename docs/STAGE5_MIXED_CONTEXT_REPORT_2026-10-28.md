# Stage 5 — Measured Mixed-Context Prefill and Generation

**Date:** 2026-10-28. **Result: 192/192 real model requests PASS**, server Qwen2.5-1.5B-Instruct FP16 under vLLM 0.19.1 on Tang RTX 4090 Laptop.

## Workload setup

- Source Git commit: `e7c96b8f9568885074630f4727e274a31cb862ed`.
- Prefix caching OFF to reduce reuse effects. Default GPU KV budget, max model length 2048, max-num-seqs 8.
- Four actual prompt lengths, 6 requests per bucket (24 per case); concurrency 1/2/4/8 and seeds 42/137.
- Each prompt length was validated against provider prompt_tokens; provider output tokens and streamed SSE TTFT are recorded. Each case passed 100% request success.

## Aggregate latency-throughput comparison

| Concurrent clients | P95 TTFT (ms), seed42 / seed137 | Output tokens/s, seed42 / seed137 |
|---:|---:|---:|
| 1 | 87.7 / 89.2 | 122.6 / 122.5 |
| 2 | 121.6 / 145.8 | 209.5 / 207.0 |
| 4 | 203.5 / 232.6 | 321.8 / 320.8 |
| 8 | 397.3 / 461.4 | 448.6 / 448.8 |

## Actual input token length and per-bucket response

| Bucket | Mean actual prompt tokens | P95 TTFT, concurrency 1 range (ms) | P95 TTFT, concurrency 8 range (ms) |
|---|---:|---:|---:|
| short | 154.0 | 19.8–30.7 | 275.9–428.4 |
| medium | 516.8 | 38.2–38.7 | 355.9–402.7 |
| long | 1209.7 | 68.9–70.8 | 277.9–370.1 |
| very_long | 1606.0 | 89.4–90.1 | 304.3–459.1 |

## Interpretation

Increasing concurrency improves aggregate throughput but worsens high-percentile TTFT on mixed contexts. Per-length TTFT need not be monotonic within an overloaded batch because short and long requests share queues and GPU work.

**Scope limits:** short deterministic completions, synthetic technical prompts repeated by design, one consumer GPU and only two seeds. No claim of production traffic coverage, formal confidence intervals or a universal KV-cache policy.

## Source evidence

- [Eight case results with all request latencies and actual tokens](evidence/stage5_kv_and_mixed/stage5_mixed_20261028).
- [Machine-generated summary with length-bucket statistics](evidence/stage5_kv_and_mixed/stage5_summary.json).
- [Pinned runtime source SHA256 manifest](evidence/stage5_kv_and_mixed/stage5_manifest.json).

## Next acceptance

Prepare heterogeneous natural document sources, output-length variation and prefill/decode scheduling alternatives. Keep each distribution frozen before comparing server settings.
