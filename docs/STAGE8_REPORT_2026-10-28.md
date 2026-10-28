# Stage 8 — Measured Multilingual Serving with Matched Arrival Rates

**2026-10-28. Result: 288/288 real vLLM SSE requests PASS.**

## Problem and controlled correction

The initial 24-request Poisson-gap experiments had unequal realized offered rates between seeds: target 8 actually gave 11.45 versus 5.93 requests/sec and target 2 gave 2.86 versus 1.48 requests/sec. These 288 requests remain valid unconditioned stochastic observations, but cannot support a matched-rate causal comparison.

The corrected experiment scales seeded exponential spacings so that the first-to-last scheduled arrival span is exactly (N-1)/target_rate. Thus both seeds have the same realized **average** offered rate while retaining relatively bursty arrivals. **The rescaled intervals are not an unconditioned homogeneous Poisson process.**

## Pinned runtime and dataset

- Source commit: 2c6a7d008a89a24af26962b7fb6c38170e0714c7. Model: vLLM 0.19.1 / Qwen2.5-1.5B-Instruct FP16 on Tang RTX 4090 Laptop GPU.
- Frozen model snapshot: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Server: loopback-only 127.0.0.1:8026, max context 2048, max-num-seqs 8, GPU memory utilization target 0.50.
- Automatic prefix caching OFF, verified in actual vLLM launch log; server log retained on Tang with SHA-256 in the manifest.
- Frozen 24 CC BY 4.0 Kubernetes technical excerpts: 8 English, 8 Simplified Chinese, 8 Japanese; four subject areas and balanced actual input-length targets around 170, 300, 470 and 680 model Tokens.
- Source corpus SHA256: d59076782e8fe177eae2b2029bef0f7ce479b86f6fc65214fb60e8e3f734b5c4. Frozen request SHA256: c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c.
- Three output instructions share max 128 output Tokens but produce real means of around 3 (label), 61 (summary) and 128 (detailed); validated by provider usage.
- Six factorial cells (3 output modes × 2 arrival rates) with two seeds each and 24 measured requests/cell/seed = 288 requests. All passed success, input Token, output Token and fixed-rate gates.

## Real observed two-seed results

The following uses the worse of the two measured P95 values and the lower observed output throughput. These are descriptive extrema, not confidence bounds. P95 from **scheduled arrival** includes waiting for a client executor worker. P95 from **actual network dispatch** excludes that earlier wait.

| Output | Matched offered rate | P95 scheduled-arrival TTFT across seeds (ms) | Worst P95 dispatch TTFT (ms) | Minimum observed output tok/s | Worst P95 client worker wait (ms) |
|---|---:|---:|---:|---:|---:|
| label | 2/s | 52.2–60.0 | 59.6 | 6.1 | 1.5 |
| label | 8/s | 53.9–54.8 | 54.2 | 23.9 | 0.7 |
| summary | 2/s | 49.8–56.4 | 56.0 | 121.1 | 1.3 |
| summary | 8/s | 65.5–68.6 | 68.2 | 420.4 | 0.7 |
| detailed | 2/s | 56.3–59.6 | 59.2 | 247.7 | 1.7 |
| detailed | 8/s | 450.8–497.5 | 65.5 | 738.4 | 452.8 |

## Engineering finding: a client-side bottleneck under heavy decoding

At detailed output and a matched 8 requests/sec, worst observed P95 scheduled-arrival-to-first-token delay was 497.5 ms, while the worst P95 TTFT once requests were actually sent was only 65.5 ms. The test client's fixed eight-worker executor had P95 queue wait as high as 452.8 ms.

This identifies **client load-generator saturation**, not measured vLLM internal scheduler waiting. A future admission-control experiment must separately record generator admission, network dispatch, HTTP response and server metrics. Otherwise it would optimize the benchmark's own thread pool rather than the inference service.

## Experimental P95 SLO choices (not an actual SLA)

Choose the highest-throughput tested offered rate among 2 and 8 requests/sec that met a proposed P95 arrival-to-first-token threshold in **both** seeds. No interpolation or production guarantee.

| Mode | P95 arrival TTFT goal | Best tested eligible offered rate | Conservative output tok/s |
|---|---:|---:|---:|
| label | 100 ms | 8/s | 23.9 |
| label | 250 ms | 8/s | 23.9 |
| label | 500 ms | 8/s | 23.9 |
| summary | 100 ms | 8/s | 420.4 |
| summary | 250 ms | 8/s | 420.4 |
| summary | 500 ms | 8/s | 420.4 |
| detailed | 100 ms | 2/s | 247.7 |
| detailed | 250 ms | 2/s | 247.7 |
| detailed | 500 ms | 8/s | 738.4 |

## Archived real evidence and first-run lesson

- [12 matched-rate raw SSE case JSON records](evidence/stage8_fixed_window/fixed).
- [Machine-generated analysis: input/output lengths, queue waits, SLO and actual rates](evidence/stage8_fixed_window/fixed_analysis.json).
- [SHA256 of every raw result, model identity, explicit cache-off server log proof](evidence/stage8_fixed_window/fixed_manifest.json).
- [Original unconditioned 12-case GPU measurement and manifest](evidence/stage8_unconditioned/unconditioned_manifest.json).
- [Original unmatched-rate exploratory report](STAGE8_EXPLORATORY_REPORT_2026-10-28.md).
- [Frozen protocol including transparent Amendment A](STAGE8_PROTOCOL.md).
- [CC BY 4.0 attribution and source Git provenance](../data/STAGE8_SOURCE_LICENSE.md).

Both 288-request experiments remain on record. The initial run is **exploratory unmatched-rate evidence**; the later fixed-duration design is a separate, controlled-rate experiment. Results are not mixed into a common pooled P95 estimate.

## Limitations and next steps

1. Two seeds × 24 requests per cell: unsuitable for high-confidence P95 SLO, confidence bounds or general performance promises.
2. Three languages are represented, but all from one cloud-native technical domain. Not representative of other industries, user queries or private enterprise documents.
3. The schedule normalizes the sample-average rate; it is not an unconditioned Poisson process or a replay of production arrivals.
4. The load injector has eight workers: waiting before dispatch can dominate and is explicitly measured separately from vLLM service TTFT. The experiment does **not** measure vLLM internal queueing in isolation.
5. The detailed-answer instruction mostly reaches the 128-token cap; long output does not demonstrate unconstrained natural stopping.
6. Real Prometheus KV utilization was sampled, but this is not a hardware-counter profile or exact per-block state trace.

**Next milestone:** build a higher-capacity, explicit open-loop arrival injector and a bounded admission/queue policy that report accepted, rejected, late, and successful requests separately. Compare real SLO violation, useful completed throughput, and dropped traffic on independent held-out arrival schedules.
