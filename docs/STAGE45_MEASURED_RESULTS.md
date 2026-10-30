# Stage45: actual dual vLLM AWQ+LoRA concurrent routing on public original user queries

**Executed on Tang single RTX4090 Laptop 16GiB:** 12 real GPU vLLM 0.19.1 dual-model server arms × **72 original user requests each = 864 offered and completed**. Both Qwen3-8B-AWQ base and rank16 SFT LoRA registered in one OpenAI-compatible vLLM model registry. Two fixed arrival seeds4511/4527 × arrival rate4 or8 offers/s × three strategies: always original base, always loaded LoRA, or Stage42 fixed prompt-only constraint classifier selecting only one model per actual request. Exactly same 72 unique original Apache OASST1 crowd prompts across arms (24/en, 24/es, 24/zh), original human query text retained locally only. Fresh GPU model server per arm, both APIs 20-token warmed, prefix cache OFF to isolate LoRA route overhead, 192 token cap and strict 72 complete+real Prometheus server counters per arm. No fake requests, no dropped rejects, source Git commit+SHA-256 immutable.

## Paired across exact user prompts, arrival sequences and 2 seeds

| Offered rate | Model choice | Mean actual useful decoded tokens/s | All-offered TTFT≤500ms | All-offered TTFT≤1000ms | P95 arrival-to-first-token | 192-token cap hit rate | Server max waiting |
|---|---|---:|---:|---:|---:|---:|---:|
| 4/s | base | 686.59 | 100.00% | 100.00% | 72.5ms | 93.1% | 0.0 |
| 4/s | lora_sft | 448.70 | 100.00% | 100.00% | 82.2ms | 37.5% | 0.0 |
| 4/s | request_router | 451.23 | 100.00% | 100.00% | 82.6ms | 39.6% | 0.0 |
| 8/s | base | 1055.05 | 37.50% | 61.81% | 1671.4ms | 93.1% | 11.5 |
| 8/s | lora_sft | 789.72 | 91.67% | 100.00% | 519.5ms | 37.5% | 5.0 |
| 8/s | request_router | 785.51 | 84.03% | 100.00% | 592.3ms | 39.6% | 5.0 |

## Actual routed-decoding performance tradeoff against always-base

| Offered rate | Router useful TPS relative to base | Router TTFT≤500ms absolute difference |
|---|---:|---:|
| 4/s | -34.28% | +0.00 percentage points |
| 8/s | -25.52% | +46.53 percentage points |

**Interpretation:** These are real 8B AWQ vLLM serving numbers, not the earlier offline NF4/BF16 Stage42 IFEval task quality. The AWQ base uses different quantization and FP16 compute and may exhibit different constraint behavior; a separate actual-server IFBench/IFEval quality audit would be required before proposing deployment. Original public OASST1 prompts were already used in Stage35/37 serving studies, not independently sampled production messages. Only **two** arrival seeds, artificial equal language quotas and 72-request windows prevent statistically established production SLO claims. Max192-token cap truncates many natural user answers (see table); high tokens/s or first-token SLO does not imply good complete responses. One engine serving both models is a functional & short-burst stress test, **not** an operator optimization proof or hour-long soak. All raw user text deliberately omitted from GitHub while hashed source and per-request numeric telemetry are published.

Source: `docs/STAGE45_PROTOCOL.md`, `scripts/run_stage45_real_dual_load.sh`, `analysis/stage45_dual_load_analysis.py`, full actual 12 cell JSONs `evidence/stage45_dual_vllm_load/`. Server logs remain on Tang with recorded SHA.
