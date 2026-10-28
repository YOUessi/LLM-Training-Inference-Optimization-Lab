# Stage16 measured diagnostics — October 29, 2026

## 8B estimator audit (retrospective)

Source: valid Stage15 AWQ 12-case archive (288 offers). Only 204 completed requests have observed latency. The existing Stage14 policy predictor was originally initialized with 1.5B timings. We must not consider it calibrated for Qwen3-8B-AWQ.

| Output mode | Completed | Median predicted ms | Median observed client latency ms | Median absolute prediction error ms | Median per-request observed/predicted |
|---|---:|---:|---:|---:|---:|
| Label | 107 | 62.092 | 417.745 | 299.291 | 4.723 |
| Detailed | 97 | 1114.325 | 2358.083 | 1322.211 | 2.251 |

These are **in-sample descriptive errors, not clean model-specific calibration**. End-to-end client latency includes service overhead and possibly server queue time; only accepted/completed requests are observed, making rejected-request effects unidentifiable. Future prediction training needs distinct train and frozen test samples and service-time components instrumented separately.

## torch.compile baseline

Real Tang RTX 4090 Laptop CUDA test, FP16 shape 128x1024; PyTorch and Triton pass correctness. Triton vs eager PyTorch: RMSNorm 3.935x, Softmax 2.167x. Against correctness-checked `torch.compile`/Inductor, **both Triton and compiled PyTorch median times round to 0.006144ms, 1.00x**. Therefore previous eager-only acceleration does NOT establish Triton superiority against compiled kernels. Small kernel timing is quantized/noisy; repeat with multiple shapes and independent sessions.

Raw JSON and checksums in `evidence/stage16`. FSDP remains `NO_SHARD` world size 1; no multi-GPU scaling claim. Remote GitHub sync is a separate status gate, not implicit in local commit creation.
