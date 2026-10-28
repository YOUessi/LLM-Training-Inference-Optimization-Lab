# Stage22 new-source domain — real Qwen3-8B-AWQ results

Source commit: `311dfe1`. 16 real GPU cases, 384 offered requests, 2 arrival seeds (9301/9309), fixed-window 8 and 16 requests/sec, 4 proxy strategies. All 16 cells passed `measured` validity checks, with matched offered request identities/arrival times within each cell, no unexplained errors, and complete SHA256 raw archives in `evidence/stage22`.

Source documents are locally installed CPython standard-library module/member docstrings licensed by the Python Software Foundation. This is a new document **source domain** relative to earlier Kubernetes and Project Gutenberg experiments. It is NOT an independently sampled natural-traffic workload, and the underlying pretraining model may already know these documents. Results are developmental only.

## Aggregated across two arrival seeds, 48 offers per policy/rate

| Rate | Policy | all-offered 250ms TTFT | Explicit rejects/48 | Mean Stage13-style U |
|---|---|---:|---:|---:|
| 8/s | FIFO400 | 50.0% | 4 | 0.6583 |
| 8/s | SJF400 | 47.9% | 4 | 0.6280 |
| 8/s | Full Adaptive | 54.2% | 7 | **0.6640** |
| 8/s | 8B calibrated | 52.1% | 7 | 0.6596 |
| 16/s | FIFO400 | 16.7% | 20 | 0.1583 |
| 16/s | SJF400 | 22.9% | 17 | 0.2263 |
| 16/s | Full Adaptive | 22.9% | 21 | 0.1994 |
| 16/s | 8B calibrated | 27.1% | 21 | **0.2286** |

U = all-offered 250ms TTFT compliance - 0.5*rejection fraction + 0.2*useful output TPS divided by matched FIFO400 TPS. This report reuses Stage13's algebra, but **does not satisfy the original candidate-win test against unlimited**, because unlimited is not among the four Stage22 arms. Do not claim full benchmark acceptance.

**Negative generalization finding:** The K8s-calibrated controller fails to dominate simpler strategies on this new stdlib domain. Its 8/s utility is slightly below uncalibrated Full Adaptive. Its 16/s advantage over SJF is only +0.0023, and it rejects 21 of 48 offered requests (>20% guardrail). Under a high offered-rate load, no policy in this set is generally strong enough to promote. 250ms TTFT may also be unrealistically tight for Qwen3-8B-AWQ. No statistically significant improvements established.

Source/original evidence: `docs/STAGE22_PRE_REGISTRATION.md`, `analysis/stage22_evaluate.py`, `evidence/stage22`, and data/checksums. Next experiment must include unlimited admission and longer duration with independent fresh workloads; it should compare 250/500/1000ms SLO curves, and account for real service-time vs queue-time separately.
