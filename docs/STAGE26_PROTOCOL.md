# Stage26 — authentic human-labeled Qwen3-8B NF4 SFT (pre-registered)

**Dataset:** original Databricks Dolly 15k (15,011 human-authored instruction/context/response pairs), official source SHA-256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`, CC BY-SA 3.0 with Databricks attribution and Wikipedia credits for relevant context. Mirrored bytes were checked against original source hash. Deterministic category-stratified samples: 512 train + 96 validation + 96 final test, 8 categories, exactly 64/12/12 examples per category. Duplicate normalized instruction+context identity excluded across all splits, sequence length (Qwen3 no-thinking chat template) <=384, no label truncation; target loss only on assistant answer tokens, all user/reference tokens masked. Split manifest is frozen before training. The dataset could be part of pretrained-model history, which restricts claims about generalization.

**Five fixed configurations, with v2 OOM amendment below (A/B/D/E 96 updates, C 192 updates):**

| ID | LoRA rank | Microbatch | Accum | Effective batch | Compute | Gradient checkpoint |
|---|---:|---:|---:|---:|---|---|
| A | 8 | 1 | 2 | 2 | BF16 | OFF |
| B | 16 | 1 | 2 | 2 | BF16 | OFF |
| C (auxiliary) | 8 | 1 | 1 | 1 | BF16 | OFF (192 steps; same approximate training samples, different updates) |
| D | 8 | 1 | 2 | 2 | BF16 | ON |
| E | 8 | 1 | 2 | 2 | FP16 | OFF |

**Evaluation:** all candidates measured on the same **validation 96** examples with answer-only cross-entropy/perplexity, measured steps 4..96 (first 3 excluded from performance timing), samples/sec, CUDA peak allocated GiB, numerical/NaN/OOM checks, exact adapter hashes. Select lowest validation assistant-token CE without seeing test. Run one separate frozen held-out test of chosen adapter over all 96 test examples (assistant-only CE/perplexity) and 12 deterministic greedy generations with reference lexical overlap (weak quality proxy, NOT proof of factual correctness or alignment). No choice is promoted if validation does not improve over base or loss unstable. Training on one GPU is QLoRA, not multi-GPU FSDP.

**Finite runner:** all GPU operations must execute via the frozen, source-pinned GPU lock executor, with safe cleanup and per-variant no-overwrite output. Failed configurations remain failed evidence, not omitted to flatter a candidate. Full 96-step trials are small development runs, not a final state-of-the-art SFT recipe. Retain train and validation separate and never use the final test for hyperparameter selection.

## Safety-preserving v2 amendment following real CUDA OOM

The frozen v1 source stopped when microbatch=2, accumulation=1, BF16, no gradient checkpointing required a 1.16 GiB allocation with only 821 MiB available (physical GPU 16 GiB). A and B had measured valid validation improvements but remain PRELIMINARY; their JSONs and the C stack trace are archived in `evidence/stage26_invalid_v1`. Frozen v2 reruns all candidates to use identical source. Variant C becomes microbatch=1, accumulation=1 for 192 optimizer steps (approximately matching A/B training-example budget, but differing optimizer-update count and effective batch). C is **throughput/memory diagnostic only**, excluded from hyperparameter selection and quality-comparison claims. A/B/D/E remain effective batch 2 and 96 steps on the original data. This is a protocol amendment, NOT a change to the original OOM evidence.
