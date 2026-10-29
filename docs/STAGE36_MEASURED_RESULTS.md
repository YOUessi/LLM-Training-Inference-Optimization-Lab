# Stage36: Qwen3-8B instruction-following on frozen Google IFEval official rule checker

**Actual measured:** exactly 96 preregistered prompts from the original 541 IFEval Apache-2.0 dataset (nine first-level instruction families, deterministic SHA ordering), three frozen Qwen3-8B NF4 models: base, Stage26 short rank16, Stage31 full512 rank16. Same original prompt, no thinking, greedy max384 output tokens, official Google `test_instruction_following_strict/loose` with original source pinned `7e672cc`. Per-prompt official booleans, generated text, SHA and throughput/response length archived in `evidence/stage36_official_ifeval/`. **Not the official full 541-prompt leaderboard result.**

| Model | Strict all-instructions prompt accuracy | Loose prompt accuracy | Strict instruction accuracy | Loose instruction accuracy | Hit 384-token cap |
|---|---:|---:|---:|---:|---:|
| base | 74.0% | 79.2% | 81.6% | 85.1% | 32.3% |
| stage26_short | 44.8% | 46.9% | 59.6% | 61.7% | 37.5% |
| stage31_full | 54.2% | 57.3% | 66.0% | 68.1% | 43.8% |

## Paired conditional uncertainty (full512 SFT minus comparator)

| Comparator | Strict prompt accuracy difference | Stratified prompt-level exploratory bootstrap 95% interval |
|---|---:|---:|
| base | -19.79 pp | [-29.17, -10.42] pp |
| stage26_short | +9.37 pp | [+0.00, +18.75] pp |

## By original IFEval first-level instruction family (strict prompt accuracy)

| Tier | Base | Stage26 short | Stage31 full |
|---|---:|---:|---:|
| change_case | 72.7% | 36.4% | 63.6% |
| combination | 63.6% | 36.4% | 45.5% |
| detectable_content | 81.8% | 54.5% | 54.5% |
| detectable_format | 72.7% | 72.7% | 72.7% |
| keywords | 54.5% | 18.2% | 9.1% |
| language | 90.9% | 63.6% | 81.8% |
| length_constraints | 60.0% | 10.0% | 20.0% |
| punctuation | 100.0% | 70.0% | 90.0% |
| startend | 70.0% | 40.0% | 50.0% |

**Research caveats:** This is only 96 examples, sampled before model generation and with publicly frozen feasibility exclusions for known long-output/NTLK sentence resource constraints. Input response truncation at 384 tokens can make some published constraint requests impossible; cap-hit rates above must be considered. Official heuristic instruction compliance is stricter/more interpretable than the earlier independent NLI semantic proxy but still does **not** establish factual accuracy, safe helpfulness or human preference. Bootstrap CI conditions on fixed public source prompts and one trained model, not multi-seed model training. No post-hoc selection based on IFEval results.

Source: `docs/STAGE36_PROTOCOL.md`, frozen `data/stage36_ifeval_96.jsonl` & SHA manifest, unmodified official evaluator under `third_party/ifeval`, full generated outputs and scoring under `evidence/stage36_official_ifeval`.
