# Stage43 — actual vLLM 0.19.1 loopback-only Qwen3-8B-AWQ base + rank16 LoRA API

**Actual executed** on Tang RTX4090 Laptop 16 GiB, real vLLM single-engine serving `qwen3-8b-awq-local` and `qwen3-8b-dolly-r16` PEFT rank16 Q/V adapter. Base+LoRA model registrations were checked from the actual OpenAI-compatible `/v1/models` endpoint, not inferred from config. **72/72 real HTTP streaming calls** finished across 24 identical public PSF-sourced technical prompts (base, LoRA, frozen Stage42 request-only router, 64 max response tokens), zero rejects, server cleanup returned zero. Raw per-request SSE TTFT and elapsed latency, model IDs, server logs, GPU memory snapshot, SHA256 and source commit in `evidence/stage43_dual_vllm/`.

| API model mode | Completed | Mean TTFT | P95 TTFT | Mean output tokens | Mean response latency |
|---|---:|---:|---:|---:|---:|
| base | 24 | 105.6ms | 130.2ms | 64.00 | 778.0ms |
| lora_sft | 24 | 130.0ms | 155.5ms | 60.75 | 878.8ms |
| router | 24 | 103.9ms | 116.6ms | 63.04 | 814.4ms |

**Do not generalize:** Model’s LoRA was trained on Qwen3-8B with **NF4/BF16** but this serving engine loads **AWQ/FP16** weights, so offline Stage42 model quality/IFEval results cannot be transplanted to this deployment without a new verification. Sequential 24 requests + 64-token output cap do not establish batch-load capacity, SLO at concurrency, full natural answer quality, statistical serving speedup, safety, or real human traffic. The router calls only one target model in deployment; the three-arm sequential comparison adds audit overhead and is not routed serving throughput. Full production-style concurrent 4/s–8/s verification is a remaining gate.

Sources: `docs/STAGE43_PROTOCOL.md`, `scripts/run_stage43_vllm_lora_server.sh`, `scripts/run_stage43_dual_vllm.sh`, raw `evidence/stage43_dual_vllm/`.
