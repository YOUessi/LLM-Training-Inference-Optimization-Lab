# Stage17 controlled diagnostics and GPU training tests

## Cross-domain prediction calibration (retrospective)

K8s FIFO400 completed requests are used only for fitting per-mode constants; Prose FIFO400 completed requests serve as evaluation. The data comes from previously used document domains and does not establish independent generalization. For 17 held-out completed labels, mean absolute client-latency error decreased 504.69 -> 264.06ms; for 17 completed detailed outputs, 1207.95 -> 382.13ms. Baseline predictions came from the prior 1.5B-initialized queue. These diagnostics estimate **client response latency**, not isolated service time, and selection on completed requests creates bias. They must not be wired into production as a service-time scheduler without instrumented queue/service decomposition and prospective evaluation.

## Six real Tang GPU MLP training configurations

FP32/BF16/FP16 autocast and accumulation 1/4 were executed for 15 optimizer steps (3 warmup, 12 measured) with effective batches 16/64 respectively. All six runs saved exact logs in `evidence/stage17`. Gradient accumulation changes the effective batch and optimization path: no controlled speedup statement is made between different effective batches. In these small synthetic matrices GPU allocation was approximately 0.095 GiB and mixed precision did not establish a reproducible memory saving. There is no full LLM fine-tuning or multi-GPU FSDP result.

| Precision | Accum | Throughput samples/s |
|---|---:|---:|
| FP32 | 1 | 18572.77 |
| FP32 | 4 | 21754.51 |
| BF16 | 1 | 16694.48 |
| BF16 | 4 | 24408.42 |
| FP16 | 1 | 15703.21 |
| FP16 | 4 | 29247.43 |

## Research gates still open

- Build instrumentation for vLLM queue wait, prefill, decode and executor overhead, with prospective calibration/train-test separation.
- Run larger held-out experiments to compare a newly calibrated scheduler rather than retroactively selecting a winner.
- Benchmark real Qwen3-8B parameter-efficient fine-tuning under matched effective batches, validating loss and OOM behavior.
- Use a multi-GPU system for genuine FULL_SHARD measurements. Single-GPU FSDP remains NO_SHARD and cannot meet this goal.
