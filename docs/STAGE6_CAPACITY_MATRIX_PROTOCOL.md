# Stage 6 — Independent KV Capacity Retention Matrix

Frozen before actual GPU execution on 2026-10-28.

## Experimental question

How do different KV GPU Block budgets change the retention of an anchor
prefix after the same number of unrelated long-prefix fills?

Cases use a fresh vLLM process for EACH independent condition:

| GPU KV Block budget | Distinct 1210-token fillers | Hypothesis |
|---:|---:|---|
| 128 | 4 | likely evicted |
| 256 | 4 | likely evicted or partial |
| 512 | 4 | more likely retained |
| 512 | 12 | likely evicted |

Same Qwen2.5-1.5B-Instruct FP16, GPU, max model length 2048,
one-request-at-a-time, seed 830, explicit prefix caching enabled.
The actual server config, real Prometheus queried/hit-token counters
and SSE TTFT/actual prompt tokens are recorded.

Every case starts with a cold anchor then immediate hit, followed by
the specified count of distinct-prefix long requests and two anchor
replays. A retained anchor **is a valid measured outcome**, not a
failure just because the original Stage5 experiment expected eviction.
Classifications are:
- evicted: <=25% hit tokens after pressure,
- retained: >=75% hit tokens after pressure,
- partial: 25–75% hit tokens after pressure.

Cold prompt <=10%, warm prompt >=75%, restored replay >=75%
are independent *validity* gates.

## Execution

  bash scripts/run_stage6_capacity_matrix.sh

The runner validates GPU idle/port free before each phase, uses separate
model processes, writes to a never-overwritten local ignored path,
records negative/failure evidence and cleans up the server it owns.

No actual GPU blocks/driver configuration is modified on disk;
the override is a vLLM process argument. One CUDA device remains in use.

The results must not be presented as direct introspection of vLLM's
replacement policy. Only inference based on cache hit-token loss is
supported. No benchmarking result is asserted in this protocol file.
