# Stage21 real LoRA asynchronous checkpoint integrity

The original Qwen2.5-1.5B LoRA training loop was extended with a bounded, one-worker asynchronous safetensors writer. A detached immutable CPU snapshot is taken synchronously; file serialization and atomic rename take place in a background thread. All asynchronous operations have an explicit `flush` barrier and propagate exceptions; no incomplete checkpoint is promoted to a final path.

On Tang RTX 4090 Laptop, 8 real training steps produced four adapter snapshots (steps 2,4,6,8). All four saved files passed SHA-256 and safetensors integrity checks, and the final 8-step asynchronous adapter was **bit-identical** to the separately executed deterministic no-checkpoint training adapter (maximum absolute parameter difference = 0). A correctness test also verifies snapshots are immutable when their source tensor changes after submission.

| Mode | Median measured optimizer-step latency | Held-out sentence loss after |
|---|---:|---:|
| No async checkpoint | 33.73 ms | 5.1258 |
| Async checkpoint every 2 steps | 40.88 ms | 5.1258 |

This measurement includes CPU snapshot transfer and potential file-serialization backpressure. It does not establish an asynchronous speedup; it establishes data integrity and exact final parameter equality on a real Transformer adapter workload. Full multi-GPU distributed asynchronous checkpointing remains outside single-GPU validation.

Frozen CPU unit tests: `tests/test_async_checkpoint.py`. Adapter evidence, per-step manifests, and original JSON: `evidence/stage21_async`.
