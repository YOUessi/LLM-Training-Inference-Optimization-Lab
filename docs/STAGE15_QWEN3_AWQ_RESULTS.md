# Stage15 Qwen3-8B-AWQ actual GPU pilot

**Source:** ba85aee. Tang NVIDIA RTX 4090 Laptop 16GB, vLLM 0.19.1, locally stored Qwen3-8B-AWQ GEMM 4-bit. Frozen Stage14 K8s input identities and arrival seed 8109 reused. 8 offered requests/s, 24 offered requests per policy, no thinking mode (`enable_thinking=false`). Results archived in `evidence/stage15_qwen3_awq/`.

| Policy | Completed / 24 | Rejected | All-offered TTFT<=250ms | Useful output tokens/s |
|---|---:|---:|---:|---:|
| FIFO Static400 | 19 | 5 | 33.33% | 271.17 |
| SJF Static400 | 19 | 5 | 45.83% | 252.48 |
| Full Adaptive | 17 | 7 | 41.67% | 257.60 |

The original 8B attempt with thinking enabled failed the short-output realization gate (completed label requests hit the 128 output-token cap). Its raw invalid JSON remains on Tang under `results/stage15_qwen3_awq_pilot/`; it is excluded from the table. The corrected run passed the driver validation. Qwen3-8B-AWQ weight loading used approximately 5.71 GiB and the vLLM startup reported approximately 5.76 GiB available KV cache at 0.82 GPU memory utilization.

**Interpretation and limits:** A single reused-document / single-arrival-seed small pilot. SJF has the best SLO250 here, while FIFO leads useful throughput and Full Adaptive rejects more requests. This is not evidence that model size alone causes the difference from 1.5B FP16: architecture, quantization and prefill/output behavior are confounded. No confidence interval or independent holdout. Do not claim Stage15 deployment superiority or generalization.

**Next:** Calibrate service times separately by prompt/output mode and model, add 8B-specific prediction-quality evaluation, run paired multi-seed tests at several offered rates, and explicitly measure error/rejection & KV pressure. Keep invalid first-run logs preserved rather than rewriting them.
