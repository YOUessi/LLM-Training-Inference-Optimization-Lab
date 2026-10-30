# Stage44 measured external IFBench60 — independent OOD check of Stage42 frozen router

**Real Qwen3-8B NF4 GPU experiment.** Official AllenAI IFBench 60 SHA-frozen test inputs, seven **non-classic IFEval** first-level precision constraint families, scored via unchanged IFBench 83-verifier package from pinned `1c40f0c` commit. Actual 60 prompt × two 8B model conditions = **120 real GPU generations** (non-thinking greedy max384 tokens), with the third routed condition choosing one of those two outputs by Stage42 fixed request-only regex **before any official scoring**. The public benchmark results below are from the original official strict and loose evaluators, not self-grading or simulated outcomes. Full private crowd user prompt and generated answer text stays on authorized Tang to avoid gratuitous republishing; GitHub retains per-prompt hashes, route decisions, full official booleans, source SHA, source code, and aggregate metrics under `evidence/stage44_external_ifbench/`.

| Frozen model condition | Strict all-constraint prompts /60 | Strict % | Loose % | Mean output tokens | Reached 384-token limit |
|---|---:|---:|---:|---:|---:|
| base | 19/60 | 31.67% | 31.67% | 254.0 | 36.7% |
| lora_sft | 7/60 | 11.67% | 15.00% | 167.0 | 25.0% |
| request_router | 18/60 | 30.00% | 30.00% | 254.2 | 36.7% |

## Frozen out-of-domain acceptance

| Stage44 preregistered criterion | Actual result |
|---|---|
| strict_at_least_base_minus_2_of60 | PASS |
| router_strict_greater_than_unscaled_sft | PASS |
| cap_rate_no_more_than_base_plus_0_05 | PASS |

**Overall IFBench out-of-domain transfer gate: PASS.** This is the first external constraint-family test after Stage42's IFEval/Dolly joint research gate; we did **not** change router weights, threshold, samples or LoRA after seeing IFBench outcomes. A failure means the Stage42 result cannot be claimed to generalize across instruction-following benchmarks.

## Matched per-family outcomes

| IFBench first instruction family | Original base strict passes | Unscaled full SFT | Router |
|---|---:|---:|---:|
| count | 6 | 2 | 6 |
| custom | 0 | 1 | 0 |
| format | 8 | 3 | 7 |
| ratio | 2 | 1 | 2 |
| repeat | 0 | 0 | 0 |
| sentence | 2 | 0 | 2 |
| words | 1 | 0 | 1 |

## Within-test family-stratified paired bootstrap, exploratory

| Comparator | Router minus comparator strict success interval |
|---|---:|
| base | [-5.00, +0.00] percentage points |
| lora_sft | [+8.33, +28.33] percentage points |

## Scope and failure modes

The test is a preselected **60/300** subset, not official full IFBench leaderboard; fixed quota sampling is not a sample of natural production traffic. Original IFBench may contain third-party model-derived data and WildChat user content under its ODC-BY-1.0 dataset licence and specific model/responsible-use terms; raw question and response text is retained locally only. The official IFBench verifier evaluates precise constraint compliance (format, words, ratios, etc.), not human factuality/safety/alignment. This experiment uses Qwen3 NF4 BF16 and cannot automatically transfer to the AWQ FP16 vLLM deployment tested in Stage43. Most importantly, a 384-token output cap may artificially lower strict success for long-output tasks, so cap-hit fractions are required in every interpretation. Only one LoRA model and 60 external examples; seed-level model training generalization is untested.

Reproducible sources: `docs/STAGE44_PROTOCOL.md`, `data/stage44_ifbench_manifest.json`, official vendored `third_party/ifbench/`, `src/llm_opt_lab/training/stage44_external_ifbench.py`, and numeric score `evidence/stage44_external_ifbench/official_scores_public.json`.
