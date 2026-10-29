# Stage29: Actual Qwen3-8B base vs selected rank16 QLoRA generations on all 96 frozen Dolly test items

Source `5fcf3ae`, single Tang 16GiB RTX4090 Laptop GPU, original Qwen3-8B NF4 checkpoint and the **validation-selected** rank16 QLoRA adapter from Stage26; identical instructions/test IDs. Qwen3 thinking disabled, deterministic greedy decode `max_new_tokens=128`, explicit attention masks; untruncated source test SHA256 `08a8334e...` matches Stage26 frozen 96-item split. All **96 paired** base/tuned generations measured and saved (per-example text SHA256 and quality metrics, original reference answers are available with proper Databricks CC BY-SA attribution in frozen dataset). The final selected B adapter SHA and source SHA are persisted in `evidence/stage29_measured/quality.json`.

## Fully paired lexical-reference metrics

| Quality metric on 96 held-out Dolly items | Base Qwen3-8B | Selected QLoRA | Difference |
|---|---:|---:|---:|
| Token-overlap F1 (one human reference) | 0.3210 | **0.4374** | **+0.1164** |
| Word ROUGE-L F1 | 0.2399 | **0.3673** | **+0.1274** |
| Normalized exact-match | 2/96 (2.08%) | 7/96 (7.29%) | +5/96 |
| Average generated words | 69.6 | 28.4 | shorter tuned outputs |
| Per-example greedy-generation wall seconds | 3.16 | 1.41 | **NOT throughput speedup**; output lengths differ |

Tuned generated answers had higher ROUGE-L for 66 examples, lower for 25, tied for 5. Classification, closed QA, open QA, summarization and brainstorming had positive ROUGE-L deltas; **creative writing declined** by approximately 0.014 ROUGE-L and 0.073 token F1, an important nonuniform result. These are weak **reference lexical** agreement metrics, do not prove factual correctness, safety, or human preference alignment. The Dolly test may overlap Qwen pretraining; no multi-seed training significance. Fixed `max_new_tokens=128` can cap detailed model answers. Faster per-query wall time mostly follows shorter generated answers, not a faster inference engine.

Original Stage29 9f35956 run failed before GPU due Bash parser syntax (`evidence/stage29_invalid_v1`); subsequent 3e83f20 run was aborted with missing explicit generation attention mask (`evidence/stage29_invalid_v2`). Both retained and excluded. Source `5fcf3ae` successfully reran the complete 96-pair test with an explicit attention mask and output fingerprint verification. No post-hoc model selection from this test occurred.

Raw 96 paired per-example metrics/sha: `evidence/stage29_measured/quality.json`. Reproduce using pinned script `scripts/run_stage29_quality.sh`; frozen source+GPU lock are mandatory.
