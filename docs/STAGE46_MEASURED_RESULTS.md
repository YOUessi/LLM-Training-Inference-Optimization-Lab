# Stage46: 120 actual AWQ vLLM API responses vs NF4/BF16 IFBench60 offline quality

**Measured:** On actual Tang vLLM 0.19.1 dual model server (loopback; AWQ/FP16 base and the NF4-trained rank16 LoRA), recorded **60 frozen IFBench prompts ×2 actual HTTP responses = 120 vLLM API calls**, deterministic thinking-disabled greedy max384, same unmodified official AllenAI IFBench strict/loose verifier from pinned `1c40f0c`. The router selected a model using only original prompt *before* generated responses; this experiment then evaluated both candidate answers for a paired counterfactual, not a three-way network batch. Raw human prompt / generated response strings remain on private Tang and only anonymized original cohort IDs/response hashes and official verdicts are committed.

## Exact paired Stage44 NF4 vs Stage46 AWQ, same original 60 IFBench source keys

| Model | NF4 strict passed | AWQ actual vLLM strict passed | Δ AWQ−NF4 | NF4-only correct | AWQ-only correct | AWQ answer cap hits |
|---|---:|---:|---:|---:|---:|---:|
| base | 19/60 | 13/60 | -10.00 pp | 6 | 0 | 40.0% |
| lora_sft | 7/60 | 9/60 | +3.33 pp | 2 | 4 | 25.0% |
| request_router | 18/60 | 12/60 | -10.00 pp | 6 | 0 | 40.0% |

**Pre-existing IFBench relative generalization gate:** NF4 offline Stage44 PASS; actual AWQ vLLM Stage46 PASS. These are preregistered within-benchmark relative criteria, **not** minimum absolute accuracy or deployment success; 60 new constraint prompts of seven families do not test all user/task populations.

## Interpretation and limitations

Stage44 NF4 and Stage46 AWQ are deliberately tested on the **same** exact IFBench60 texts/constraints, allowing direct paired quantization/backend compatibility verification; their agreement is not evidence of another independent new dataset. Different decoding responses and cap hits are real quantization/backend effects and need further investigation. Stage45 separately measured 864 concurrent API responses under actual dual vLLM but was primarily a serving latency/load experiment on multilingual OASST1 source, not a correctness checker. Strict IFBench is still low absolute, output caps remain material. No proof of human-rated helpfulness, safe complete answers, hour-scale traffic performance, real customer task performance, or multi-GPU training. Production rollout BLOCKED.

Source: `docs/STAGE46_PROTOCOL.md`, `scripts/run_stage46_awq_ifbench.sh`, `src/llm_opt_lab/serving/stage46_vllm_ifeval_quality.py`, `evidence/stage46_awq_ifbench_quality/` and original NF4 `evidence/stage44_external_ifbench/`.
