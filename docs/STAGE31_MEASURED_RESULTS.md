# Stage31: real Qwen3-8B NF4 QLoRA — all 512 Dolly training records used twice

**Actual measured**, source ff7171da, Tang RTX 4090 Laptop GPU 16GiB. Four fully measured independent rank/seed conditions, each 512 AdamW updates across two complete passes over the same 512 human-written CC BY-SA 3.0 Dolly training records (1,024 reference exposures), 96 validation records at base, epoch 1, epoch 2. **All four best checkpoints are from epoch 1**; epoch 2 validation CE increases for every rank/seed (a possible overfitting signal on this small instruction subset; not definitive generalization evidence). Training run logs and hashes are retained with the raw JSON in `evidence/stage31_full512`. All four adapters were retained in the frozen Tang result folder, but only the validation-selected adapter is published as an actual model-weight file on GitHub.

| Rank/seed | Epoch1 validation assistant CE | Epoch2 validation assistant CE | Selected epoch | Training samples/s | Peak PyTorch allocated GiB |
|---|---:|---:|---:|---:|---:|
| rank16_seed3111 | 1.7809 | 1.7956 | 1 | 4.48 | 12.93 |
| rank16_seed3127 | 1.7770 | 1.7985 | 1 | 4.47 | 12.93 |
| rank8_seed3111 | 1.7827 | 1.7901 | 1 | 4.51 | 12.87 |
| rank8_seed3127 | 1.7802 | 1.7882 | 1 | 4.48 | 12.87 |

Selected by **validation only**: `rank16_seed3127` (rank 16, seed 3127, epoch 1, validation CE 1.77700). Frozen original 96-item test selected **once**, assistant-only CE base **2.81454** vs full-SFT **1.61445**, perplexity 16.685 → **5.025**. Prior Stage26 short-run 96-test CE 1.63689, so current new full-run selected test CE improves slightly (~0.02244), but the same 96 test IDs were already observed in Stage26 research; Stage34 provides a genuinely previously unused *record* subset.

The small differences between Rank8 and Rank16 are **not** statistically established with only two random seeds each. A best epoch1 vs epoch2 observation is repeatable here but the public Dolly collection may overlap model pretraining. Source/adapter hashes and per-epoch metrics are machine-verifiable in the original JSONs; all rank/seed weights are not claimed to be on GitHub. No multi-GPU FSDP or human alignment benefit is implied.

