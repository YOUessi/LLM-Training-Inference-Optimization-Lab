# Stage14 K8s 8101 pilot — real GPU, development-only

Source SHA `b4cc118`, Qwen2.5-1.5B-Instruct FP16 served via local vLLM on Tang RTX 4090 Laptop; existing K8s documents, new arrival seed 8101, offered rate 16 requests/second. Exactly the same 24 offered request identities and scheduled offsets in both arms. Real SSE results are archived with SHA256 in `evidence/stage14_pilot_k8s_seed8101/`.

| Metric | FIFO Static400 | Full Adaptive |
|---|---:|---:|
| Offered | 24 | 24 |
| Completed | 22 | 21 |
| Rejected | 2 | 3 |
| All-offered TTFT <=250ms | 75.0% | 83.33% |
| Useful output tokens/s | 561.45 | 530.66 |
| Client admission queue P95 ms | 338.1 | 209.1 |

The full strategy improves this local 250ms coverage but has worse rejection and useful throughput; single small developmental cell is not evidence of a general win. This pilot is not one of the pre-registered full eight-policy independent holdout cells. The logger prints `STAGE13` inherited from Stage13, but the structured record `kind` is `stage14_development_mixed_vllm_ablation`, with Stage14 source commit. Do not use the printed prefix to misclassify it.
