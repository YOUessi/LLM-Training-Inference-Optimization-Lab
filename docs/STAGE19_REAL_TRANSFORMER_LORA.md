# Stage19 genuine Transformer adapter training (not synthetic MLP)

On Tang RTX 4090 Laptop 16GiB, the offline Qwen2.5-1.5B-Instruct checkpoint was loaded in BF16 with frozen base weights. A manual rank-8, alpha-16 LoRA implementation wrapped 56 Q/V projection linear modules, training exactly 1,089,536 adapter parameters. Unlike the earlier FSDP or synthetic MLP demonstrations, these are actual causal-LM forward/backward passes. Adapter checkpoints are safetensors, with independent SHA256 and key/shape checks.

| Accumulation | Measured steps | Median optimizer-step ms | GPU peak allocated GiB | Holdout loss before | After |
|---|---:|---:|---:|---:|---:|
| 1 | 6 | 33.73 | 2.970 | 5.2020 | 5.1258 |
| 4 | 6 | 139.22 | 2.974 | 5.2020 | 4.9915 |

Both runs use tiny, repeated educational text samples and only one held-out sentence: these changes are smoke evidence, NOT proof of fine-tuning quality, generalization, or fair training throughput; effective batch sizes differ. A first attempt using Tang's system Python failed due to old Pillow lacking `Image.Resampling`; switching to the already isolated `llm-opt-serving` environment resolved it without modifying system packages.

Raw metrics, trainable adapter checkpoints, SHA256 manifest, and reproducibility tests are in `evidence/stage19_lora/` and `tests/test_stage19_lora_evidence.py`.

Stage20 aims to validate Qwen3-8B NF4 QLoRA in a *separate* environment; no result is implied by the existence of source code.
