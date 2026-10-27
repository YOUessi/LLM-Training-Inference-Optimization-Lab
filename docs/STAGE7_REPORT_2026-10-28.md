# Stage 7 — Natural Prose Prefill and Mixed Short/Long Decode

**Date:** 2026-10-28. **Outcome: 384/384 actual model SSE requests passed**, across 16 combinations of four concurrency levels, two output modes and two random seeds.

## Data and reproducibility

- Model and hardware: vLLM 0.19.1, Qwen2.5-1.5B-Instruct FP16, NVIDIA RTX 4090 Laptop GPU; Torch 2.10.0+cu128, driver 580.178.04.
- Input corpus: 30 frozen non-overlapping excerpts sampled from five English public-domain-in-USA works (social fiction, science fiction, natural science, travel fiction, fantasy); exact sources and SHA256 recorded in data/.
- Real GPU experiment Git commit: `00993045f8564539428ce65c5a9170f5c840ab1d`. Analyzer source commit: `38d32600b7f6e74f08b290bd11a198f7bf6e5f34`.
- Corpus exact SHA256: `2912a4d8cc17789a1b6b339cac927a27ec41f24f7de140d84466ecd441e095ca`.
- Prefix caching explicitly OFF (confirmed in actual service launch log). Loopback 127.0.0.1:8026, max_model_len 2048, max-num-seqs 8, GPU memory utilization 0.50.
- Every condition uses 24 unique frozen passages (6 per length, all five genres). Identical seed reuses EXACT same text/bucket assignments between concurrency and output modes. Seed 42 uses forward order; seed 137 uses reverse concurrency order.
- Every output JSON includes real provider prompt_tokens, completion_tokens, TTFT, estimated TPOT, source genre/document ID, prompt hash, latency and throughput.
- No generated content stored in evidence; raw server log stays on Tang with SHA256 in manifest.

## Input lengths (real provider tokens)

| Input group | Short-mode observed prompt tokens | Long-mode observed prompt tokens | Preregistered acceptance range |
|---|---:|---:|---:|
| short | 161–225 | 169–233 | 110–240 |
| medium | 402–522 | 410–530 | 345–580 |
| long | 816–1051 | 824–1059 | 740–1140 |
| very_long | 1203–1630 | 1211–1638 | 1120–1720 |

**The input lengths include both excerpt content and response instructions.** The four bucket acceptance checks passed in every case, with 100% prompt and completion token usage.

## Actual decoding and latency-throughput Pareto

The short-answer mode generated about 30–31 output tokens/request with max_new_tokens=32; long mode actually generated 128 tokens/request with max_new_tokens=128. Both satisfy preregistered actual-output length gates. The model mostly reaches the token cap, so these tests compare **two fixed output budgets**, not a broad real-world natural stopping-length distribution.

| Mode | Concurrency | P95 TTFT range across seeds (ms) | Output token throughput range (tok/s) | Mean emitted output tokens |
|---|---:|---:|---:|---:|
| short | 1 | 83.0–83.1 | 123.9–125.7 | 30.8 |
| short | 2 | 102.0–113.7 | 207.0–214.1 | 30.6 |
| short | 4 | 178.4–240.1 | 326.0–332.8 | 30.6 |
| short | 8 | 287.4–375.1 | 462.9–476.9 | 30.6 |
| long | 1 | 82.5–83.3 | 141.5–142.0 | 128.0 |
| long | 2 | 105.5–138.2 | 270.4–271.4 | 128.0 |
| long | 4 | 208.8–254.5 | 484.8–486.8 | 128.0 |
| long | 8 | 302.5–389.4 | 827.4–829.6 | 128.0 |

## Exploratory P95 TTFT SLO decisions (derived analysis, not a new GPU run)

Decision rule: for each mode and concurrency, take the **worse of two observed P95 TTFTs** and the **lower of two measured token throughputs**; choose the highest-throughput concurrency still meeting a proposed SLO. These are descriptive two-seed extrema, **NOT confidence bounds or guaranteed production SLOs**. Thresholds are exploratory post-hoc values, not preregistered operational commitments.

| Proposed P95 TTFT limit | Short mode: selected concurrency / conservative tok/s | Long mode: selected concurrency / conservative tok/s |
|---|---:|---:|
| 100 ms | 1 / 123.9 | 1 / 141.5 |
| 150 ms | 2 / 207.0 | 2 / 270.4 |
| 250 ms | 4 / 326.0 | 2 / 270.4 |
| 400 ms | 8 / 462.9 | 8 / 827.4 |
| 600 ms | 8 / 462.9 | 8 / 827.4 |
| 1000 ms | 8 / 462.9 | 8 / 827.4 |

At the experimental 250 ms limit, short output reaches concurrency 4 while long output only qualifies at concurrency 2 (the worse long-output concurrency 4 P95 was slightly above 250 ms). **Do not extrapolate a hard threshold** from two runs or treat the difference as a causal inference about long decoding alone: prompt instructions and emitted token lengths both changed.

## Failure discovered, preserved and resolved before successful rerun

- Initial source commit `e214c39777632a74fd6497bafeb7291d51aa9233` was scientifically **INVALID**, not a completed 384-request validation. Four short-mode cases passed, but first long-mode case had short-bucket 207–269 input tokens, above its preregistered upper bound 240.
- Root cause: the verbose long-answer instruction added prefill prompt tokens. Simply raising max_new_tokens did not explain this; the prompt itself increased.
- The original input acceptance limits were retained. A concise directive requesting 25 numbered observations was Qwen-tokenizer-calibrated first (short 169–233 tokens), then the entire 384-request matrix was executed afresh under source `00993045f8564539428ce65c5a9170f5c840ab1d`.
- The initial invalid 5 raw case JSON files, a failure summary, and their original SHA256 hashes were retained; none were counted among the 384 corrected requests.

## Detailed machine evidence

- [Sixteen corrected raw real GPU SSE case reports](evidence/stage7_natural/stage7_corrected).
- [Frozen analysis of all 384 requests and matched two-seed SLO tables](evidence/stage7_natural/stage7_analysis.json).
- [Original failed run: four short cases and one invalid long case](evidence/stage7_natural/stage7_invalid_first_run).
- [Failed protocol root-cause summary](evidence/stage7_natural/stage7_initial_invalid_summary.json).
- [Source commits, model, server flags, server log SHA and SHA256 for every original JSON file](evidence/stage7_natural/stage7_source_manifest.json).
- [Corpus rights and exact source attribution](../data/README.md).
- [Original protocol and transparent Amendment A](STAGE7_PROTOCOL.md).

## Scientific limitations

1. Source excerpts are real published English prose but drawn from 5 old works, not real contemporary production user requests, private documents, or multilingual traffic.
2. Only two seed repetitions and 24 requests per case; bucket-specific P95 is estimated from only 6 requests per length. Not reliable enough for high-confidence SLO guarantees.
3. The output cap of 32/128 tokens is generally reached; next test must include varied genuine stopping lengths and arrival rates.
4. Input distributions changed compared with Stage5 synthetic repeated-paragraph benchmark, so between-stage differences should be interpreted as **distribution shift**, not as proof of a new kernel or inference-system performance optimization.
5. Cache explicitly OFF. No cache eviction, PagedAttention ablation, or GPU SM/DRAM counters are measured in this stage.
6. Additional real-world text domains and non-English source traffic remain to be tested; do not claim the five public-domain texts constitute a fully representative corpus.

## Next milestones

- Stage8: freeze an independently sampled modern multi-domain and multilingual corpus with explicit redistribution rights, evaluate varied decoding stop-lengths and concurrent arrivals.
- Explore SLO-aware admission/queueing policies only against that held-out workload, with repeatable interval estimates and a correctness/no-loss gate.
- Keep multi-GPU FSDP and Nsight Compute occupancy/DRAM metrics BLOCKED until actual additional GPU or authorized profiler permissions become available.
