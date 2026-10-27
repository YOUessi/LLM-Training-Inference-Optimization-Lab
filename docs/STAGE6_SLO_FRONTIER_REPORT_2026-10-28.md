# Stage 6 — P95 TTFT SLO Versus Generated Throughput Frontier

**Date:** 2026-10-28. This is a *new analysis module*, not a new GPU benchmark. It reuses the eight real vLLM Stage5 mixed-context files collected on Tang RTX 4090 Laptop GPU.

## Workload, numerical basis and decision rule

- Real server: vLLM 0.19.1, Qwen2.5-1.5B-Instruct FP16, prefix caching disabled.
- Four input length buckets (approximately 154, 517, 1209 and 1606 input tokens), equal six-request composition per concurrency, 24 SSE requests per seed.
- Four tested concurrency levels 1/2/4/8, two frozen seeds 42/137; total 192 requests. All provider prompt/output token usage was complete.
- For each tested concurrency, use the **larger** of its two observed P95 TTFT values and **smaller** of its two observed output-token throughput values. This is a conservative two-run descriptor, *not a statistical confidence bound*.
- For a given P95 TTFT target, choose the observed concurrency maximizing conservative throughput among those whose worst observed P95 TTFT is at/below the target.

| P95 TTFT SLO | Highest qualified concurrency | Minimum observed output tokens/s across the two seeds | Worst observed P95 TTFT (ms) |
|---:|---:|---:|---:|
| 100 ms | 1 | 122.5 | 89.2 |
| 150 ms | 2 | 207.0 | 145.8 |
| 250 ms | 4 | 320.8 | 232.6 |
| 500 ms | 8 | 448.6 | 461.4 |

## Observed Pareto-style design points

| Concurrency | P95 TTFT range across seeds (ms) | Completion token throughput range (tok/s) |
|---:|---:|---:|
| 1 | 87.7–89.2 | 122.5–122.6 |
| 2 | 121.6–145.8 | 207.0–209.5 |
| 4 | 203.5–232.6 | 320.8–321.8 |
| 8 | 397.3–461.4 | 448.6–448.8 |

## Interpretation

If the experimental target is P95 TTFT <=150 ms, concurrency 2 is the largest tested setting meeting it on both runs; if the target is <=250 ms, concurrency 4 qualifies. Under 500 ms, all four tested levels qualify and concurrency 8 gives the highest observed throughput.

This is **not a global concurrency auto-tuner**. It uses two seeds, synthetic repeated technical input, one consumer GPU, fixed model and output cap, and only the four explicitly measured concurrency values. Running it on different prompts, model revisions, GPUs or KV budgets requires fresh live evidence.

## Reproduction and evidence

- [Derived machine-readable SLO analysis](evidence/stage6_slo_frontier.json).
- [Original Stage5 mixed-context experiment and raw per-request traces](evidence/stage5_kv_and_mixed/stage5_summary.json).
- [Stage5 full mixed-context report](STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md).
- Reproduce locally without GPU from the frozen CSV-free JSON data using the new source module:

    PYTHONPATH=src python -m llm_opt_lab.serving.slo_frontier --evidence-dir docs/evidence/stage5_kv_and_mixed/stage5_mixed_20261028 --out /tmp/stage6_slo.json

## Next Stage6 GPU gates

1. Replace synthetic repeated documents with a frozen, varied natural text corpus.
2. Measure prefix-cache eviction under 128/256/512 block budgets and multiple filler counts with fresh-server independence.
3. Repeat SLO frontier estimation for longer decode lengths and mixed request arrival rates.
4. Keep Nsight Compute hardware counters and two-device FSDP sharding BLOCKED until actual authorized resources exist.
