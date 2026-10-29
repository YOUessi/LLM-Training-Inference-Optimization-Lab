# Stage26 Qwen3-8B real human-labeled SFT and locked test results

**Dolly 15k, CC BY-SA 3.0, verified upstream SHA256**, eight categories; deterministic 512 train + 96 validation + 96 frozen test. Model Qwen3-8B actual 4-bit NF4 QLoRA on single Tang RTX 4090 Laptop; assistant-answer-token-only cross-entropy, all user and reference-context tokens masked, Qwen3 no-thinking chat template, max full sequence 384 tokens. All five original v2 configurations run over real data and measured GPU, no artificial generated answers. Configs A/B/D/E have 96 optimizer updates and same effective batch 2; auxiliary C has 192 updates with batch1 (approximately same exposed sample count), and is **not eligible for controlled quality selection**. Source SHA pinned in result `source_sha`, raw metrics/logs under `evidence/stage26_sft_v2`.

## Frozen validation (96 examples, answer-only metrics)

| ID | Configuration | Validation assistant CE | Samples/s | PyTorch peak allocated GiB |
|---|---|---:|---:|---:|
| A | rank8 BF16, accum2, no GC | 1.80770 | 4.638 | 12.718 |
| B | **rank16 BF16, accum2, no GC** | **1.79943** | 4.600 | 12.783 |
| C (auxiliary) | rank8 BF16, accum1, 192 steps | 1.79158 | 4.589 | 12.704 |
| D | rank8 BF16, accum2, **GC ON** | 1.80785 | 3.389 | **10.028** |
| E | rank8 **FP16**, accum2, no GC | 1.80754 | 4.602 | 12.719 |

Base validation assistant CE before SFT was ~2.95254 BF16. Within the same 96-update/effective-batch-2 candidates A/B/D/E, B has the lowest validation CE and is selected **without using final test labels**. The validation gap B vs A (0.0083 nats/token) is small; do not claim a statistically significant rank gain without repeated seeds. GC saves ~2.69 GiB (~21.2% of observed allocation) at ~27% lower samples/s on this short-sequence regime; GC can be worthwhile when memory is the binding constraint. C has lower CE but twice the optimizer update count and is excluded from fair selection.

## Single-use final test of selected B (96 never-trained-on records)

| Test metric | Untuned base | Selected rank16 QLoRA |
|---|---:|---:|
| Assistant cross-entropy nats/token | 2.81454 | **1.63689** |
| Assistant perplexity | 16.6855 | **5.1392** |

Twelve deterministic generated examples had a mean reference word-overlap F1 ~0.347. This weak lexical metric is **not a factual-correctness, safety or human-preference score**; source answers could overlap model pretraining. No benchmark-state-of-the-art or general alignment claim. QLoRA was trained on 192 selected training examples across 96 updates (of 512 available); a longer independent run is needed for serious post-training work.

### Preserved negative evidence

Stage26 v1 tried microbatch2 without GC and OOMed on an allocation request of ~1.16 GiB with ~821MiB free on a 16GiB GPU. Original A/B metrics and the C traceback remain archived in `evidence/stage26_invalid_v1`; they are **excluded** from final v2 comparison. The v2 worktree reran all candidates from a new frozen SHA with a clearly documented auxiliary C protocol amendment. Only selected B PEFT adapter (including config and SHA256) is committed to the repository to avoid five redundant large model blobs; the other four artifact hashes and full source configs are stored, with their original checkpoints retained on Tang for reproducibility.

Follow-up: more training steps with full 512-record train split, larger independent test, task-level semantic/format correctness, and a controlled instruction-following evaluation beyond reference perplexity. Full multi-GPU FSDP remains impossible on the current one-GPU host.
