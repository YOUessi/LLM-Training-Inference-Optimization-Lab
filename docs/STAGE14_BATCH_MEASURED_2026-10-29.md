# Stage14 32-case GPU batch — development results

Frozen experiment source: `7d24eae`. Tang RTX 4090 Laptop GPU, vLLM 0.19.1, Qwen2.5-1.5B-Instruct FP16, prefix caching OFF.

32 cases × 24 offered requests = 768, 18 rejected, zero unexplained errors. Four matched domain/rate cells, eight policies. Raw per-request JSON and SHA256 archive in `evidence/stage14_batch_20261029/`. Previously seen document corpora; this is **not independent holdout**.

| Domain | Rate/s | Policy | SLO250 all-offered | Rejected | Useful output tokens/s |
|---|---:|---|---:|---:|---:|
| k8s | 8 | fifo_dynamic | 100.0% | 0 | 413.6 |
| k8s | 8 | fifo_static400 | 100.0% | 0 | 414.2 |
| k8s | 8 | fifo_static400_ewma | 100.0% | 0 | 413.8 |
| k8s | 8 | full_adaptive | 100.0% | 0 | 413.9 |
| k8s | 8 | sjf_aging_static400 | 100.0% | 0 | 413.8 |
| k8s | 8 | sjf_dynamic | 100.0% | 0 | 413.8 |
| k8s | 8 | sjf_static400 | 100.0% | 0 | 414.2 |
| k8s | 8 | sjf_static400_noewma | 100.0% | 0 | 414.2 |
| k8s | 16 | fifo_dynamic | 83.3% | 4 | 578.1 |
| k8s | 16 | fifo_static400 | 83.3% | 0 | 571.1 |
| k8s | 16 | fifo_static400_ewma | 83.3% | 0 | 570.7 |
| k8s | 16 | full_adaptive | 79.2% | 4 | 576.4 |
| k8s | 16 | sjf_aging_static400 | 83.3% | 0 | 577.5 |
| k8s | 16 | sjf_dynamic | 79.2% | 4 | 574.5 |
| k8s | 16 | sjf_static400 | 91.7% | 1 | 571.3 |
| k8s | 16 | sjf_static400_noewma | 91.7% | 1 | 572.0 |
| prose | 8 | fifo_dynamic | 100.0% | 0 | 404.4 |
| prose | 8 | fifo_static400 | 100.0% | 0 | 404.2 |
| prose | 8 | fifo_static400_ewma | 100.0% | 0 | 404.8 |
| prose | 8 | full_adaptive | 100.0% | 0 | 404.8 |
| prose | 8 | sjf_aging_static400 | 100.0% | 0 | 404.5 |
| prose | 8 | sjf_dynamic | 100.0% | 0 | 404.8 |
| prose | 8 | sjf_static400 | 100.0% | 0 | 404.7 |
| prose | 8 | sjf_static400_noewma | 100.0% | 0 | 404.4 |
| prose | 16 | fifo_dynamic | 91.7% | 2 | 610.2 |
| prose | 16 | fifo_static400 | 95.8% | 0 | 604.4 |
| prose | 16 | fifo_static400_ewma | 95.8% | 0 | 606.7 |
| prose | 16 | full_adaptive | 95.8% | 1 | 612.0 |
| prose | 16 | sjf_aging_static400 | 95.8% | 0 | 609.2 |
| prose | 16 | sjf_dynamic | 95.8% | 1 | 611.6 |
| prose | 16 | sjf_static400 | 95.8% | 0 | 608.8 |
| prose | 16 | sjf_static400_noewma | 95.8% | 0 | 608.8 |

## Interpretation
At 8 offers/s all eight policies achieved 100% SLO250 in both domains. At 16 offers/s, SJF static400 achieved 91.7% on K8s while full adaptive achieved 79.2%; K8s FIFO static400 was 83.3%. Prose SLO250 was 95.8% for most policies. The full adaptive mechanism did not show a robust benefit. A single seed per domain/rate cannot establish statistical significance or generalization. Policy order was fixed and may confound comparison with load/thermal drift.

## Next gates
Independent unseen documents and additional paired seeds; randomized/counterbalanced policy order; multiple request bursts; paired confidence intervals; per-mode reject accounting; longer high-concurrency traces; and CUDA/training tracks.
