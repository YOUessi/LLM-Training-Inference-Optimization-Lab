# Stage 8 — Unconditioned Poisson arrivals: valid exploratory benchmark, unmatched realized rates

**Experiment 2026-10-28. Real vLLM 0.19.1 and Qwen2.5-1.5B-Instruct FP16 on Tang RTX 4090 Laptop. 288/288 actual SSE requests PASS.**

## Frozen licensed corpus and rate caveat

- 24 pinned Kubernetes CC BY 4.0 technical excerpts: English 8, Chinese 8, Japanese 8, four cloud-native technical topics.
- Real Qwen-token-calibrated context targets approximately 170/300/470/680 prompt tokens, six documents per length, repeated identically across modes/rates within a seed.
- Three response instructions use the SAME 128-token cap but realize label ~3 tokens, summary 57.5-61 and detailed 128 on average.
- Source commit: 8a6053a1d4ee8fa3850dd362a9ebb6ab9039b7f8; analyzer commit: 09273883f9afc9d81518cd28b99902773fafefaa.
- **Important:** the 24 sampled exponential gaps are not conditioned on elapsed duration. The actual average offered rates deviate significantly from the targets, so the two seeds cannot be interpreted as matched arrival rates.

| Mode | Target req/s | Seed 42 actual req/s | Seed 137 actual req/s | Seed42 P95 scheduled TTFT | Seed137 P95 scheduled TTFT |
|---|---:|---:|---:|---:|---:|
| label | 2 | 2.86 | 1.48 | 55.1 ms | 54.0 ms |
| label | 8 | 11.45 | 5.93 | 57.1 ms | 58.5 ms |
| summary | 2 | 2.86 | 1.48 | 56.8 ms | 52.6 ms |
| summary | 8 | 11.45 | 5.93 | 75.0 ms | 66.2 ms |
| detailed | 2 | 2.86 | 1.48 | 66.1 ms | 56.1 ms |
| detailed | 8 | 11.45 | 5.93 | 930.3 ms | 195.8 ms |

## What the measurements demonstrate (and do not)

Under high offered arrival load, detailed decoding creates substantial client-executor queue wait. In one seed at target rate 8, P95 from *planned arrival* reached 930.3 ms while actual-send TTFT was 64.1 ms. This must not be labeled vLLM internal queue latency: the client has only 8 workers and records their own waiting separately.

The mean target is not the observed rate in a short random window. For target 8 req/s the realized samples were about 11.45 and 5.93 req/s. Thus a causal statement such as “seed alone caused the 930-vs-196 ms difference at the same arrival rate” is NOT justified.

The Stage8 fixed-window randomized-spacing follow-up conditions the total schedule duration to the target mean rate, preserves random relative gaps but is NOT an unconditioned homogeneous Poisson process. It will be a distinct experiment and cannot overwrite this exploratory result.

## Raw evidence

- [Twelve machine-produced GPU request logs](evidence/stage8_unconditioned/unconditioned).
- [Independent full results and mode/rate summary](evidence/stage8_unconditioned/unconditioned_analysis.json).
- [Machine source, model, original file SHA256, actual rate confound](evidence/stage8_unconditioned/unconditioned_manifest.json).
- [Frozen scientific protocol and transparent Amendment A](STAGE8_PROTOCOL.md).

**Limits:** one GPU, 2 seeds and 24 requests per cell, fixed technical-document corpus and model; no formal P95 SLO confidence interval, dynamic scheduling/queue policy or model-quality evaluation.
