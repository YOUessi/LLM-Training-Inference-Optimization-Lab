# Stage 10 — Transparent Client-Side Admission: True SLO Gain vs Rejection Cost

**2026-10-28. Real GPU measurement: 192 offered requests, 186 completed and 6 explicitly rejected; zero unexpected errors.**

## The bottleneck and policy

Stage9 demonstrated that merely increasing client worker threads migrates waiting to vLLM, barely changing total scheduled-arrival P95 TTFT. In this experiment, the baseline sends all requests immediately with 32 client threads. The candidate **client-side admission proxy**, not a vLLM scheduler change, keeps at most eight in-flight requests, four local waiting slots and a 250 ms admission deadline. Overflow/expiry is explicitly rejected and counted.

## Reproducible controls

- Experiment source SHA: 1f654f7404a7cc7e1cd5a88f28fc0615349a154f. Independent analysis source SHA: 7b4618b5f1a1d66fe3168d69763bde77ac0856af.
- Model vLLM 0.19.1, locally cached Qwen2.5-1.5B-Instruct FP16, NVIDIA RTX4090 Laptop GPU, vLLM max-num-seqs 8, context length 2048, GPU utilization target 0.5, server cache OFF.
- Frozen source input SHA256: c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c; identical 24 multilingual Kubernetes documentation prompts and identical seeded fixed-window arrivals for each matched pair. These conditioned random arrival gaps are not an unconditioned Poisson process.
- Offered load: **8 requests/sec**. Two output behaviors (short resource label, long detailed analysis), two seeded orders (42 and 137), two policies. Eight real cases × 24 offers = 192 requests.
- Same 32-thread load generator for both policies, so a client worker-count bottleneck is no longer the experimental treatment.
- All-offered-request SLO means completed responses that emitted their first Token within 250 ms of scheduled arrival **divided by all offered requests**, including every rejected/failed request. Completion/rejection is tracked separately.
- All actual completions passed provider prompt/output token and mode-length checks, with no unexpected network/model errors. vLLM Prometheus running/waiting and KV usage gauges were sampled.

## Matched results from real model requests

| Mode | Seed | Policy | Offered | Completed | Explicit rejects | All-offered TTFT <=250 ms | Useful output Tokens/s |
|---|---:|---|---:|---:|---:|---:|---:|
| label | 42 | unlimited | 24 | 24 | 0 | 100.0% | 24.3 |
| label | 42 | bounded | 24 | 24 | 0 | 100.0% | 24.4 |
| label | 137 | unlimited | 24 | 24 | 0 | 100.0% | 23.9 |
| label | 137 | bounded | 24 | 24 | 0 | 100.0% | 23.9 |
| detailed | 42 | unlimited | 24 | 24 | 0 | 75.0% | 747.2 |
| detailed | 42 | bounded | 24 | 21 | 3 | 87.5% | 709.3 |
| detailed | 137 | unlimited | 24 | 24 | 0 | 66.7% | 769.6 |
| detailed | 137 | bounded | 24 | 21 | 3 | 87.5% | 709.5 |

## Does the admission control actually help?

- **Prompt first-token timeliness:** In detailed mode, baseline completed 34/48 offers within 250 ms (70.8%), while bounded admission completed 42/48 (87.5%). Every rejected offer was counted as noncompliant, so the improvement in this specific timely-first-token fraction is **16.7 percentage points**.
- **Rejected traffic:** The bounded policy explicitly refused 3 out of 24 detailed offers in each seed, **6/48 (12.5%)**. All were accounted for; no status or model response was silently dropped.
- **Useful throughput cost:** In the detailed mode, baseline output throughput was 747.2 and 769.6 tokens/s across two seeds, versus 709.3 and 709.5 tokens/s with bounded admission. Thus the SLO compliance improvement came with lower useful output throughput and fewer served requests.
- **Negative control:** For the label mode, both policies completed all 24/24 per seed, rejected none, and had 100% timely first Tokens. Limiting client admission offers no demonstrated advantage at this light workload.

**Conclusion:** bounded client admission improves the fraction of all offered detailed requests receiving their first Token by 250 ms in this finite workload, but sacrifices successful request coverage and output throughput. It is **a measured tradeoff, not a universal net serving optimization**.

## Do not conflate latency metrics

Accepted-only P95 TTFT is conditional on success and can be made misleadingly low by rejecting slow arrivals. Whenever any offer is rejected, the exact P95 latency of *all offered requests* is unobserved and saved as null. The report instead uses explicit all-offered SLO compliance plus rejection and useful throughput.

Admission deadline is defined by waiting for client in-flight capacity, not a guarantee that the entire model response finishes by 250 ms. SLO measures time to **first Token**, not full completion.

## Original evidence

- [All eight real admission/control SSE request logs](evidence/stage10_admission/real_sse).
- [Strict independently recomputed four-pair analysis and language-level request outcomes](evidence/stage10_admission/stage10_analysis.json).
- [Pinned model, client/sever settings, SHA256 of every raw file and source revisions](evidence/stage10_admission/stage10_manifest.json).
- [Preregistered admission policy and evaluation contract](STAGE10_PROTOCOL.md).
- [Stage9 precondition: real queue-location A/B](STAGE9_REPORT_2026-10-28.md).

## Scientific and engineering limits

1. This is **client-side admission** using semaphores, not a new vLLM internal scheduler or server implementation.
2. The queue of semaphore waiters is not strict FIFO. There is no fairness guarantee; per-language rejection patterns from just 8 offers/language/case must not be generalized.
3. There are only 24 offered requests per condition and two seeds; no confidence intervals or claims of production SLO guarantees.
4. Requests come from one cloud-native Kubernetes documentation domain and fixed prompt/output instructions, not production heterogeneous or multilingual user traces.
5. The fixed-window schedule has randomized spacing but is not an unconditioned homogeneous Poisson process. Server was not restarted between all policy conditions (order was reversed by seed).
6. Real Prometheus gauges were sampled at 150 ms, which may miss instantaneous queue peaks. Hardware occupancy counters and extra GPUs remain unavailable.
7. Improving one service metric by rejecting 12.5% of requests may not be desirable; future policies must compare request utility, business cost and fairness, not just first-token timing.

## Next acceptance gate

Test more admission deadlines and independent high-burst/low-burst schedules, report **all-offered** compliance/throughput/fairness jointly, and compare a true server-side scheduler only if implemented and validated. Collect sufficiently long traces or repeated held-out data before claiming a robust policy improvement.
