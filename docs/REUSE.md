# Reuse / attribution ledger

## Existing repositories (pinned inspection 2026-10-26)

1. [YOUessi/agentic-gpu-debugger](https://github.com/YOUessi/agentic-gpu-debugger), `main@33a6f230061d11dcf96410d7bc37e2d8d2bdbea9`.
   - Reused **methodology only**: CPU/oracle validation, reproducible evidence, fail-closed status, toolchain manifests.
   - Not reused: CUDA debugging Agent, Compute Sanitizer report format, patch/repair engine, V2 CLI, code or binaries.
   - Important: V2 PROJECT_SPEC_CN_V2 explicitly excludes Nsight profiling and kernel performance tuning.

2. [YOUessi/CodeRL-Lab](https://github.com/YOUessi/CodeRL-Lab), `main@556c8035198b2482e0b803ae6b169209ef6e32c8`.
   - Reused **methodology only**: pinned configs, fixed candidate budgets, confidence intervals and fair experimental comparison.
   - Not reused: TRL GRPOTrainer implementation, training datasets, RL reward code or model weights.
   - Distributed-training benchmark here is an independent DDP reference implementation, not a claim about the old training pipeline.

## Source ownership and proof policy

All initial lab source files are independently authored. No existing repository file has been copied into this scaffold. If code is later copied, record: original URL, full SHA, exact paths, license, changes, tests, and measured outcomes **before** merging.

Two GitHub repositories remain unchanged. A GitHub repo for this lab requires creation if the connected tooling lacks repository-creation permission; until then, this is a local Git repository.
