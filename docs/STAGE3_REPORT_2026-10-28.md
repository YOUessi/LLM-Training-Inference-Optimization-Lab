# Stage 3 — Verified real vLLM serving and corrected long-output workload

Date: 2026-10-28. All measured values below originate from the pinned, checked-in JSON files linked in this document. Streaming client uses provider-reported completion tokens (not SSE chunk counts).

## Environment and traceability

- Host: Tang / NVIDIA RTX 4090 Laptop GPU (16 GiB physical memory), NVIDIA driver 580.178.04.
- vLLM version 0.19.1, PyTorch 2.10.0+cu128, Triton 3.6.0; FP16 Qwen2.5-1.5B-Instruct.
- Cached model snapshot: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- vLLM configured to bind 127.0.0.1:8026 only, max model context 2048, max_num_seqs 8, GPU memory utilization target 0.50.
- Official PyPI vLLM wheel checksum was verified before installation (SHA256 71a87f46cafab4489c69a5c5c83b870d0235e5694d8222303d460576293dc719).
- Git source at initial live test: ff37cd5; first extended experiment: 6f5dcde; corrected output-length gate and repeated benchmark: 2749782.
- vLLM engine, model and SSE benchmark client are separate processes. No user-facing remote port; server was terminated after each suite.

## Initial real vLLM baseline (8 requests per concurrency)

| Concurrent clients | P95 TTFT (ms) | P95 client-estimated TPOT (ms) | Completion token throughput (tokens/s) |
|---:|---:|---:|---:|
| 1 | 14.97 | 6.72 | 141.84 |
| 2 | 20.01 | 6.72 | 275.32 |
| 4 | 24.28 | 7.00 | 529.76 |

All initial measured requests succeeded (24/24) and usage coverage was 100%.
Raw machine records: [initial summary](evidence/stage3_vllm_summary.json) with raw client files stage3_vllm_concurrency_1/2/4.json.

## Extended 32-token-budget short-answer test (24 requests per concurrency, each measured request emitted 21 tokens)

Two sequential sweep orders (ascending and descending) use different random seeds, with 2 warmup requests per case. The generated answers all stopped at 21 tokens, below the 32-token cap.

| Concurrency | Forward tokens/s | Reverse tokens/s | P95 TTFT forward (ms) | P95 TTFT reverse (ms) |
|---:|---:|---:|---:|---:|
| 1 | 141.8 | 141.8 | 15.2 | 14.5 |
| 2 | 283.2 | 282.3 | 14.8 | 17.6 |
| 4 | 530.2 | 527.6 | 25.5 | 25.6 |
| 8 | 1005.5 | 1019.9 | 30.2 | 28.4 |

All 192/192 measured short-answer requests succeeded; token-usage coverage was 100% in all eight cases. These two independent sweep orders reduce order artifacts but do not establish population confidence intervals.
Raw reports: [forward summary](evidence/stage3_vllm_repeated/short_forward/summary.json) and [reverse summary](evidence/stage3_vllm_repeated/short_reverse/summary.json), alongside all 8 original per-concurrency JSON files.

## Corrected true long-decode workload

An initial 128-token CAP experiment still emitted only 21 tokens, because the benchmark client hardcoded a one-sentence answer instruction. That test is explicitly excluded from long-output results and retained as a negative outcome: [invalid-long summary](evidence/stage3_vllm_invalid_long/summary.json).

Correction: the request now asks for at least 25 numbered technical considerations, and the benchmark gate requires measured mean completion tokens >=90 for max_tokens=128. The validated workload emitted exactly 128 tokens per request.

| Concurrency | P95 TTFT (ms) | P95 total request latency (ms) | P95 client-estimated TPOT (ms) | Output tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 14.8 | 866.6 | 6.71 | 147.8 |
| 4 | 23.2 | 898.1 | 6.91 | 571.2 |
| 8 | 29.2 | 913.3 | 6.96 | 847.3 |

All 36/36 long-decode requests succeeded; provider token usage was available for every request; the actual mean 128-token output length was verified rather than inferred from the cap.
Raw reports: [corrected long summary](evidence/stage3_vllm_repeated/long_decode/summary.json) and concurrency 1/4/8 per-request evidence in the same directory.

## Comparison with the serialized Transformers server

The Stage 2 Transformers service used a global generation lock (no continuous batching). The Stage 3 initial vLLM baseline used the same offline Qwen model and eight requests per concurrency, with the same short-answer prompt template and 32-token budget. These are independent runs, not an isolated PagedAttention ablation.

| Engine at concurrency 4 | P95 TTFT (ms) | Output tokens/s |
|---|---:|---:|
| Serialized Transformers | 601.5 | 109.5 |
| Real vLLM 0.19.1 | 24.3 | 529.8 |

Observed throughput ratio: 4.84x. P95 TTFT ratio: 24.8x. These are workload-specific results, not a universal acceleration guarantee.

## Accuracy, safety and known limits

- TTFT is time until the first nonempty streamed text fragment; TPOT is estimated from end-to-end wall time minus TTFT divided by provider-reported remaining output token count. Neither is a direct CUDA per-token timer.
- Warmups are excluded from measured requests; the three workload suites use fixed per-request template, temperature=0, seed and pinned CUDA/driver/model.
- The service samples are finite, the synthetic prompt set is intentionally narrow, and input prefix reuse and cache effects can affect results. Real production traffic, varying context lengths and throughput SLOs need separate tests.
- Long output reached the configured 128-token cap. That proves actual longer decoding, not a general model quality improvement.
- Installing vLLM only modified the lab Python environment. Pip-check reported a conflict between a shared MediaPipe installation and lab Protobuf; this unrelated package was not modified globally.
- Nsight Compute/Systems and genuine multi-GPU sharded FSDP metrics are not part of this serving experiment.

## Remaining engineering gates

1. Change prompt length and request distribution to test long-context prefill/cache behavior and throughput/latency Pareto curves.
2. Add GPU hardware profiling with a version compatible with R580 and ADA (SM8.9); document perf-counter permissions.
3. Run two-device full sharding only if two real GPUs are available, otherwise keep BLOCKED.
