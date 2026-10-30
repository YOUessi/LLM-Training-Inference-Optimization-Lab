# Stage42 — prompt-only hard-constraint routing, new IFEval96 and Dolly128 ID-disjoint tests

**Real measured on Tang RTX4090 Laptop 16GiB**: original Qwen3-8B NF4 vs frozen Stage31 full512 rank16 QLoRA vs a lightweight deterministic prompt-only rule switching to the original model on explicit format/count/negative/structural constraints. This is a **handcrafted inference-time specialist router, not a new trained model**. The request decision uses *only* user prompt text and frozen regex weights, no official benchmark label, reference answer or generated text. For auditing, both original and LoRA counterfactual responses are generated and scored offline before routing metrics are computed, but the actual router requires only one model forward per request. Exactly 96 **new** IFEval source keys excluding 325 previously observed ones, and 128 **new** human-written Dolly source identities excluding previous 960 selected items, all frozen before GPU run. Original source SHA, licensed source, per-case model output SHA, official Google rule scores and full answer-only CE are in `evidence/stage42_router/`.

## Previously unseen Google IFEval96 subset: official strict instruction compliance

| Frozen model choice | Strict passed /96 | Strict prompt accuracy | Loose prompt accuracy |
|---|---:|---:|---:|
| base | 75/96 | 78.12% | 81.25% |
| lora_sft | 63/96 | 65.62% | 66.67% |
| request_router | 73/96 | 76.04% | 80.21% |

## New Dolly128 reference answer-token CE (same test IDs and target tokens)

| Frozen model choice | CE (nats / target token) | Perplexity |
|---|---:|---:|
| base | 2.86518 | 17.5522 |
| lora_sft | 1.66722 | 5.2974 |
| request_router | 1.87809 | 6.5410 |

## Router trace and research go/no-go
- IFEval route counts: {'base': 86, 'lora_sft': 10} (hard-constraint priority to base).
- Dolly route counts: {'lora_sft': 104, 'base': 24} (ordinary request priority to LoRA).
- Predeclared joint research gate: **PASS**; conditions {'near_base_ifeval_allow_2_prompts': True, 'dolly_CE_improves_at_least_0_50_nats': True, 'zero_benchmark_label_or_answer_in_router': True, 'completed_source96_and128': True}. Gate requires official IFEval router successes within 2/96 of original base AND Dolly128 answer CE at least 0.50 nats/token better than original base, with no official rule-checker output as routing input.

## Within-source stratified case bootstrap (exploratory only)

| Comparison | IFEval strict accuracy delta (router minus comparator) | Dolly answer CE delta (router minus comparator) |
|---|---:|---:|
| base | [-5.21, +0.00] pp | [-1.1284, -0.8723] nats/token |
| lora_sft | [+0.00, +20.83] pp | [+0.1387, +0.2911] nats/token |

## Explicit limitations / non-production status

One pretrained model and one LoRA checkpoint, public IFEval and Dolly *within-source* previously-unseen identities (potential pretrained knowledge overlap), and a lexical constraint router whose correct classification on benchmark-format instructions may be trivial. This is an intervention evaluating a baseline *engineering risk-routing rule*, **not** a learned novel algorithm or proof it generalizes to implicit, long-context, multilingual user constraints. The baseline and LoRA counterfactuals were generated sequentially for fair score verification; true deployed model routing would call one backend only and needs a separate latency/SLO load study. IFEval strict checker measures programmatic constraint obedience, not factual helpfulness or safety. Conditional fixed-case bootstrap is not true multi-seed population inference. Production authorization remains blocked until genuine user quality/error rates, end-to-end vLLM dual-backend routing and larger varied tasks are verified.

Sources: `docs/STAGE42_PROTOCOL.md`, `src/llm_opt_lab/training/stage42_risk_router.py`, frozen `data/stage42_manifest.json`, machine source `evidence/stage42_router/frozen_router_result.json`.
