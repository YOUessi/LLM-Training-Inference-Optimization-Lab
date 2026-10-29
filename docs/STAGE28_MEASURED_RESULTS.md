# Stage28: real pretrained Transformer optimizer-state async checkpoint and recovery

Run Qwen2.5-1.5B-Instruct BF16, frozen base and rank8 Q/V LoRA on a single Tang RTX4090 Laptop GPU, **12 actual forward/backward/AdamW steps** in each of four deterministic runs: no-checkpoint continuous; synchronous checkpoint each 3 steps; asynchronous disk writing each 3 steps; and a **fresh model+optimizer process-equivalent reinitialization after step6**, restoring adapters, optimizer moments, CPU/CUDA RNG and data-generator RNG from a SHA-256 verified safe `torch.load(weights_only=True)` state. Source `e0df0b1` and test commands pinned in `scripts/run_stage28_batch.sh`; raw evidence and manifests in `evidence/stage28_measured`.

| Actual Transformer training mode | Median measured step ms | Samples/sec | Notes |
|---|---:|---:|---|
| No checkpoint continuous | 56.93 | 17.15 | CPU-snapshot overhead absent |
| Synchronous full-state checkpoint | 60.06 | 14.55 | CPU transfer and serialization on training thread |
| Asynchronous disk writing | 62.48 | 14.61 | CPU snapshot still synchronous, 5.2–5.6 ms submission copy |
| Restored from async checkpoint at step6 | 59.69 | 16.35 | Exact final adapter, optimizer hash and loss |

**All four runs produced identical final adapter SHA, optimizer-state SHA and final loss `4.6167335510`.** Ten checkpoint files passed SHA-256 with atomic manifests; the two representative step6 checkpoints plus one final step12 checkpoint, full metadata, and all ten hashes are published. Complete original ten-checkpoint 127MiB archive is retained on Tang under the immutable experiment results; publication of three does not pretend the entire archive is on GitHub.

This establishes exact deterministic training recovery and no silent checkpoint omission, **not an asynchronous checkpoint throughput win**. Asynchronous saved transfer cost in the background but actual step median is slightly worse than continuous, and similar to synchronous in this tiny workload. A single GPU cannot validate distributed FSDP sharding, network storage, or multi-rank collective recovery; do not claim those capabilities.
