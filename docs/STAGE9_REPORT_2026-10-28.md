# Stage 9 — Where Did Queue Latency Go? Real GPU 8 vs 32 Client Workers

**2026-10-28, 288/288 actual model SSE requests succeeded.** The first controlled experiment in Stage9 holds server settings and the offered request schedule fixed while varying only the number of load-generator client threads.

## Problem addressed

Stage8 matched-rate detailed-response experiments at 8 offered requests/sec experienced approximately 450–500 ms P95 first-token delay measured from scheduled arrival, despite roughly 60 ms P95 TTFT after the network request was sent. The load generator had eight worker threads, and it was unknown whether the excessive delay originated at the client, inside vLLM, or both.

## Preregistered controls

- Exact real GPU experiment source SHA: 9f497193c38c8d2bb00b6b78fd25d6b07bc66528. Analyzer source SHA: cc58db358f4605fc9632707b5a30c5dbffe6de2e.
- Qwen2.5-1.5B-Instruct FP16 on Tang NVIDIA RTX 4090 Laptop GPU, vLLM 0.19.1.
- Same frozen 24 Kubernetes CC BY 4.0 technical excerpt requests, three source languages, four calibrated input lengths; input SHA c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c.
- Explicit vLLM prefix caching OFF, context 2048, GPU-memory-utilization target 0.5, server max-num-seqs=8, one loopback service, real server log SHA recorded in manifest.
- Exact same **fixed-window**, seeded randomized arrival times for each paired condition; equal empirical offered rates. This schedule is NOT an unconditioned Poisson process.
- Six matched pairs: seeds 42/137 × conditions label@8 req/s, detailed@2 req/s and detailed@8 req/s, with client executor capacity 8 versus 32 workers.
- 24 real requests per arm/case = 288 real requests, all 100% success, real provider prompt/output Token usage and output-length gates passed.
- Real vLLM Prometheus waiting and running requests gauges sampled every 150 ms and preserved as summary metrics (samples may miss short spikes).

## Real measured results

Client P95 wait means scheduled arrival to availability of a client thread. Server P95 TTFT means time from actual network dispatch until first nonempty SSE content fragment. Arrival P95 TTFT includes both waiting portions. **Quantiles of separate stages must not be added arithmetically**.

| Workload | Seed | Workers | P95 client wait (ms) | P95 dispatch TTFT (ms) | P95 scheduled-arrival TTFT (ms) | Max sampled vLLM waiting requests | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| label @ 8/s | 42 | 8 | 0.7 | 53.6 | 54.1 | 0 | 24.3 |
| label @ 8/s | 42 | 32 | 0.6 | 53.1 | 53.7 | 0 | 24.3 |
| label @ 8/s | 137 | 8 | 0.8 | 52.5 | 52.8 | 0 | 23.9 |
| label @ 8/s | 137 | 32 | 0.7 | 53.1 | 53.7 | 0 | 24.0 |
| detailed @ 2/s | 42 | 8 | 1.4 | 55.4 | 55.9 | 0 | 247.8 |
| detailed @ 2/s | 42 | 32 | 1.4 | 53.8 | 54.1 | 0 | 247.7 |
| detailed @ 2/s | 137 | 8 | 1.6 | 57.6 | 58.1 | 0 | 248.1 |
| detailed @ 2/s | 137 | 32 | 1.6 | 53.5 | 54.1 | 0 | 248.1 |
| detailed @ 8/s | 42 | 8 | 452.2 | 66.1 | 517.8 | 0 | 739.3 |
| detailed @ 8/s | 42 | 32 | 0.7 | 509.3 | 509.7 | 4 | 742.0 |
| detailed @ 8/s | 137 | 8 | 393.2 | 57.8 | 451.8 | 0 | 764.5 |
| detailed @ 8/s | 137 | 32 | 0.8 | 435.4 | 435.9 | 4 | 767.2 |

## Paired differences: 8 versus 32 client workers

| Workload | Seed | Reduction in P95 client wait (ms) | Reduction in P95 total arrival TTFT (ms) | Change in P95 dispatch TTFT (ms) | Change in peak sampled vLLM waiting requests |
|---|---:|---:|---:|---:|---:|
| label @ 8/s | 42 | 0.1 | 0.5 | -0.4 | +0 |
| label @ 8/s | 137 | 0.1 | -0.8 | +0.6 | +0 |
| detailed @ 2/s | 42 | 0.1 | 1.8 | -1.6 | +0 |
| detailed @ 2/s | 137 | 0.0 | 4.0 | -4.1 | +0 |
| detailed @ 8/s | 42 | 451.5 | 8.1 | +443.1 | +4 |
| detailed @ 8/s | 137 | 392.4 | 15.8 | +377.6 | +4 |

## Finding: moving the bottleneck is not removing it

- At detailed 8 req/s, 8-client-thread P95 queue wait reached **393–452 ms**, while send-to-first-token P95 stayed around 58–66 ms. The client prevented many requests from reaching the server promptly.
- At 32 threads, client executor wait dropped to about **0.7–0.8 ms**, but P95 TTFT after actual dispatch rose to **435–509 ms**. The real vLLM gauge recorded up to **4 waiting requests**, compared with zero sampled at the eight-thread setting.
- Total P95 scheduled-arrival-to-first-token decreased only **8.1 ms and 15.8 ms** across the two seeds, far less than the 392–452 ms decrease in client wait.
- Label@8 and detailed@2 are negative controls: both worker configurations had near-zero client queue, and their timing differences were small. This supports the interpretation that overload, not thread count itself, activates the queue.

**Therefore raising client workers is not an inference optimization.** It shifts waiting toward the actual vLLM queue, with only a small net impact on the overall first-token tail under this fixed load.

## Evidence and reproduction

- [Six paired real worker-ablation results computed from all 288 actual requests](evidence/stage9_worker_isolation/stage9_analysis.json).
- [All 12 raw vLLM SSE request files with actual token usage and per-request timings](evidence/stage9_worker_isolation/real_sse).
- [GPU/model/source commits, exact raw file SHA256, server-cache flag confirmation](evidence/stage9_worker_isolation/stage9_manifest.json).
- [Frozen Stage9 experiment protocol](STAGE9_PROTOCOL.md).
- [Stage8 matched-rate baseline and full 288-request experimental record](STAGE8_REPORT_2026-10-28.md).

## Limits

1. Two seeded, 24-request trials per condition: tail values are exploratory descriptive statistics, not confidence bounds or a certified SLO.
2. vLLM waiting gauges sampled every 150 ms can miss microbursts; zero sampled queue is not proof that no request ever waited.
3. No new vLLM scheduler or admission controller was implemented in this experiment; only client submission worker capacity changed.
4. The workload is one cloud-native technical domain in three languages; not representative of general enterprise traffic.
5. Matched tests were run sequentially on one GPU; residual caching, thermal and clock effects may remain even with reverse condition order.
6. The server accepted all requests and none was rejected in this experiment, so there is no evidence yet on admission policy or the usefulness/fairness tradeoff of dropping requests.

## Next acceptance gate — transparent admission policy

Implement an admission queue with a bounded number of in-flight server requests, explicit queue size, waiting deadlines and honest reject/timeout reporting; compare using identical **offered** (not merely accepted) traffic. The SLO denominator must include all offered requests, and total useful completed output throughput must be reported alongside the accepted-only latency.
