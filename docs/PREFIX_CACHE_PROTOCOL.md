# Stage 4 — Prefix Cache Ablation Protocol

Date: 2026-10-28. Frozen **before** actual GPU runs.

## Hypothesis and factorial design

Repeated long-prefix requests may gain TTFT and throughput when vLLM automatic
prefix caching is ENABLED, with otherwise equal model/hardware/workload. Measure
cache ON vs OFF and shared vs distinct prefixes independently.

| Server cache arm | Request prefix arm | Expected reusable prefix? |
|---|---|---|
| Explicit enabled | Shared long document + unique task suffix | Yes |
| Explicit disabled | Shared long document + unique task suffix | No |
| Explicit enabled | Unique first-content ID + same long document | Minimal |
| Explicit disabled | Unique first-content ID + same long document | No |

Two independent seeds (42 and 137), reversed condition order for seed 137.
Concurrency 1 and 8, 24 measured requests and 2 warmups per case.
Total planned = 2 cache states × 2 modes × 2 concurrencies × 2 seeds ×
24 = 384 real measured SSE requests.

## Fixed controls and data validity

- Qwen2.5-1.5B-Instruct FP16, one RTX4090 Laptop GPU, vLLM 0.19.1.
- Server loopback-only port 8026, max_model_len 2048, max_num_seqs 8,
  GPU memory utilization target 0.50.
- Same fixed 35-repetition technical context, different marker early
  in first content tokens for the distinct arm.
- Real input token counts from provider prompt_tokens, bounded [900,1450].
  Missing input usage or length outside bounds FAIL the case.
- Real completion_tokens from SSE. TTFT = first nonempty content SSE,
  TPOT is an end-to-end client estimate, not CUDA per-token time.
- All experimental conditions require 100% request success.
- Server starts with explicit enable/no-enable-prefix-caching flag;
  metric snapshots before/after each case are archived on Tang.
- Separate vLLM server for caching off and caching on to avoid
  configuration carry-over; warmups populate any reusable prefix.
- Reversing order by seed reduces, but cannot eliminate, thermal/load drift.

## Running

Run on Tang with idle GPU, with the experiment script:

    bash scripts/run_prefix_cache_ablation.sh

Raw logs/metrics/results are retained under the ignored local path
results/prefix_cache_ablation_20261028. No unrelated GPU process,
system package or existing repo is modified. The runner cleans up
the server it starts and refuses existing output directories.

## Scientific claim limits

Match results by (prefix mode, seed, concurrency) across cache ON/OFF.
Use the distinct-prefix negative control, inspect server prefix cache
metric deltas, and do not infer causal attribution from a single
uncontrolled workload. This intentionally synthetic, repetition-heavy
corpus is not representative of production inputs.

No speedup has been asserted here before real measurement.

## Corrected negative-control protocol (revision 2)

The first full experiment completed 384 valid HTTP requests, but half of
cache-enabled distinct-prefix cases unexpectedly hit >99% reused tokens.
Cause: the negative-control marker varied with (seed,index) only, so
**concurrency 1 and 8 under the same seed replayed identical prompt
IDs**. On the second execution, the previous case's cached prefix was
eligible for perfect reuse. This is cross-*case* reuse, not a vLLM bug.

Corrected marker uses (seed + 1009 * concurrency, index) for the distinct
arm only, so it changes between concurrency conditions as well as within
them. Shared-prefix cases intentionally keep the same text.

Additionally, the runner now gates actual vLLM Prometheus prefix-token
metrics between each case: cache OFF -> zero queries/hits; cache ON,
shared -> hit_fraction >=70%; cache ON, distinct -> hit_fraction <=10%.
No result is accepted without the metric checks; any failed case is
preserved under ignored results and must be reported, not reinterpreted.
