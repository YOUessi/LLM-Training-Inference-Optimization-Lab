# Stage18 calibrated proxy real GPU development pilot

Source commit d29d959, Qwen3-8B-AWQ on Tang RTX 4090 Laptop vLLM 0.19.1, thinking disabled. 4 real SSE cells, 96 offers, identical 24 frozen prose requests and paired scheduled arrivals for each offered rate. Arrival seed 8209 not previously used in this Stage15 train domain; content domain reused so this is **not independent corpus validation**. The 8B proxy calibration constants were fitted retrospectively on K8s completed FIFO requests; response latency is not decomposed GPU service time.

| Rate | Policy | Completed/24 | Rejected | All-offered 250ms TTFT | Useful tokens/s |
|---|---|---:|---:|---:|---:|
| 8/s | Full Adaptive | 18 | 6 | 45.83% | 276.2 |
| 8/s | 8B-calibrated | 18 | 6 | 50.00% | 275.6 |
| 16/s | Full Adaptive | 14 | 10 | 16.67% | 318.1 |
| 16/s | 8B-calibrated | 14 | 10 | 25.00% | 324.3 |

Under these matched pairs the calibrated heuristic increases timely-response counts by one at 8/s and two at 16/s, without changing accepted/rejected totals. No statistical significance claim is warranted from 24 offers per cell; sequential GPU policy testing may introduce time/order drift. Further seeds, counterbalanced order, independent content domains and direct service-time instrumentation are mandatory before promotion. No production scheduler enabled.

Raw data and SHA256 manifest: `evidence/stage18`.
