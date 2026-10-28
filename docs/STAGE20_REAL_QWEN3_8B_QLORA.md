# Stage20 — REAL Qwen3-8B single-GPU NF4 QLoRA

Tang has a single NVIDIA RTX 4090 Laptop 16GiB GPU. The original local Qwen3-8B BF16 checkpoint was loaded into a fresh Python virtual environment with bitsandbytes NF4 double quantization. `peft` trained rank-8/alpha-16 LoRA Q/V adapters over 36 decoder layers (144 A/B adapter tensors, 3,833,856 trainable parameters). Both configurations ran actual model forward, backward and optimizer steps on a small self-contained educational corpus. The independent adapter reload was successful and its held-out loss differed from the recorded post-train loss by only 0.01264.

| Actual Qwen3-8B metric | Accum=1 | Accum=2 |
|---|---:|---:|
| Optimizer steps total | 5 | 5 |
| Measured steps after warmup | 3 | 3 |
| Median optimizer-step time | 151.61 ms | 303.79 ms |
| Samples/sec | 6.59 | 6.59 |
| Max PyTorch allocated VRAM | 8.256 GiB | 8.271 GiB |
| Adapter parameters | 3,833,856 | 3,833,856 |
| One-sentence held-out loss before | 6.9517 | 6.9517 |
| One-sentence held-out loss after | 6.9107 | 6.9001 |

The two different effective batch sizes yield different optimization trajectories and cannot establish higher quality or an acceleration. The tiny repeated training corpus and single held-out sentence are engineering smoke checks, **not alignment-quality validation**. The 8B base model itself is NOT committed (the user has it locally); the two 15MB PEFT adapters, configs, SHA256 and metrics are saved under `evidence/stage20_qlora/`.

Isolated environment package versions: `torch==2.10.0+cu128`, `transformers==4.57.6`, `safetensors==0.8.0`, `peft==0.18.1`, `accelerate==1.15.0`, `bitsandbytes==0.49.2`, `Pillow==11.3.0`. The base model was loaded offline from `/home/you/models/Qwen3-8B`. This was a separate venv; existing vLLM serving environment was not changed. One actual independent GPU reload (accum1 adapter) yielded loss 6.92331 vs 6.91067 logged after training (absolute drift 0.01264). The upload does not imply proprietary trained model weights or task-specific quality improvement.

Source: `src/llm_opt_lab/training/qwen3_8b_qlora_smoke.py`, `verify_qwen3_qlora.py`; tests `tests/test_stage20_qlora_evidence.py`.
