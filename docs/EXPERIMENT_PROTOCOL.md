# Measurement protocol

1. Pin SHA, Python, PyTorch/CUDA/Triton, GPU model, driver and tool versions.
2. Isolate test suites; no source copied from legacy repositories without a provenance entry.
3. Pin tensor shapes, dtypes, strides and semantics; compare with a numerically appropriate reference.
4. Warm up kernels and serving endpoints before recording timings, then synchronize CUDA.
5. Log each independent experiment with its own JSON; keep negative results.
6. Report median and P95 latency and raw values, not only relative speedup.
7. For vLLM use streamed nonempty text output for TTFT, and server **usage.completion_tokens** for TPOT. Never estimate true tokens from SSE chunk count.
8. CUDA instrumentation must use real Nsight tools and supported GPU; do not use dummy profiling screenshots.
9. Track regressions and variability (shape sweeps, repeat runs, confidence intervals where possible).
10. If a required tool or GPU is missing, record `blocked` with reason and do not fabricate benchmark output.

11. When comparing against torch.compile, ensure the final dtype cast is inside the compiled function (otherwise an extra eager kernel artificially inflates the baseline).
