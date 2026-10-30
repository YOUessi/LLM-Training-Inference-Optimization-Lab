# Stage41: new 128 Dolly instructions never selected in any previous 832 source identities

**Actual real Qwen3-8B NF4 GPU evaluation**. Freeze an independent-by-record CC BY-SA Databricks Dolly128 holdout (16 each from eight original categories), excluding all previously selected Stage26/31 train/val/test 704 and Stage34 separate128 (832 excluded) BEFORE the Stage38/Stage40 inference-time and format-SFT selection results. No new model/adapter/strength tuning based on this data. Source and selected decision SHA validated. Strictly identical assistant-answer target tokens / input messages for four preselected variants: unmodified base, old Stage31 rank16 SFT, Stage38 guarded residual α (may be α0 fallback), Stage40 constraint-format replay (may be base fallback). Real reference CE and perplexity only; not semantic factual correctness.

| Pre-selected model | Untouched Dolly examples | Answer CE nats/token | PPL |
|---|---:|---:|---:|
| base | 128 | 2.97840 | 19.6564 |
| original_full | 128 | 1.74027 | 5.6989 |
| guarded_strength | 128 | 1.75139 | 5.7626 |
| format_replay | 128 | 2.97840 | 19.6564 |

**Previous dev choices (not from this test):** Stage38 residual α=0.75; Stage40 format-training candidate `base`. Alpha=0 or selected='base' legitimately means no adapted model passed its guardrails; do **not** relabel such fallback as a successful new adapted model.

## Stratified fixed-sample paired CE differences versus original base
| Model | Within-source 95% bootstrap interval (CE new − original base) |
|---|---:|
| original_full | [-1.41049, -1.10390] nats/token |
| guarded_strength | [-1.39683, -1.09261] nats/token |
| format_replay | [+0.00000, +0.00000] nats/token |

**Limitations:** The 128 sample identities are genuinely disjoint from previous 832 locally selected Dolly items, but Dolly is still the SAME publicly available upstream source and Qwen pretraining may have encountered it. Conditional bootstrap only resamples records and does not capture independent training seeds, models or natural human task distributions. Lower reference loss does not excuse a loss of strict IFEval instruction obedience (see Stage38/40 separate unseen tests) or certify safety/semantic correctness. Data license and source hashes are in `data/stage41_manifest.json`; full numeric per-case CE and SHA checks are in `evidence/stage41_new_dolly128/`.
