# Stage 9 — Client Executor vs Server Queue: Frozen Ablation

Preregistered **before** GPU execution on 2026-10-28.

## Problem

Stage8 real vLLM detailed requests at 8 offered req/s produced ~451–498 ms
P95 from scheduled arrival to first Token, while send-to-first-Token
TTFT stayed ~58–66 ms. The Stage8 client had only eight worker threads.
Before changing server scheduler/admission logic, isolate this load
generator artifact.

This is **not yet a custom server admission policy experiment**.

## Two-arm hardware/serving design

- Same model and source: Qwen2.5-1.5B-Instruct FP16 cached locally.
- Same Tang NVIDIA RTX 4090 Laptop GPU, vLLM 0.19.1.
- Same server max-num-seqs 8, max context 2048, GPU memory target 0.5.
- Prefix caching explicitly OFF; loopback-only endpoint 127.0.0.1:8026.
- Same 24 frozen CC BY 4.0 Kubernetes input prompts per case; eight en,
  eight zh-cn and eight ja, across four approximately token-balanced lengths.
- Identical **fixed-window** seeded exponential spacing, NOT an
  unconditioned homogeneous Poisson process. Realized average offered
  rate is exactly the target for both seeds.
- Client executor worker count: **8 (previous Stage8 load-generator cap)
  versus 32 (greater than all 24 requests per case)**.
- 2 seeds (42,137), reverse worker and test order to reduce order artifacts.
- 3 conditions: label @ 8 req/s, detailed @ 2 req/s,
  detailed @ 8 req/s; 24 measured requests each.
- 2 worker arms × 2 seeds × 3 conditions × 24 = **288 real requests**.
  One warmup each case is excluded.

## Metrics / safety requirements

Every request: original scheduled arrival; client execution start; real
SSE first Token; actual prompt_tokens/completion_tokens; latency;
source language; prompt SHA; real offered rate; output token/s.

For each case: count/success/error, client thread executor P95 wait,
P95 from scheduled arrival, P95 from actual dispatch, output tok/s.
New real vLLM Prometheus periodic gauges (every 150 ms):
num_requests_waiting, num_requests_running and KV cache usage;
missing gauges must be unavailable/null, not converted to zero.

**Fail closed:** 24/24 requests must succeed; actual input/output usage
complete; all four language/context groups match the frozen request list;
mode completion gate satisfied; offered realized rate exactly 2 or 8;
worker count actually in log; server cache-off from actual server launch.

No hardcoded expected speedup or automatic "PASS" when 32 workers are
faster. The scientific conclusion is allowed to be negative.

## Interpretation

If raising workers to 32 reduces client queue wait but raises server
queue wait or post-send TTFT, the experiment isolates *where* waiting
occurs. It does NOT by itself improve vLLM's internal scheduler. Total
scheduled-arrival latency and service completions must be tracked to
avoid relocating delay and claiming an optimization.

Server gauges sampled at 150ms can miss short-lived spikes. P95 from
24 requests with two seeds is descriptive and not a reliable SLA.

## Actual execution

On Tang when GPU is idle:

    bash scripts/run_stage9_worker_isolation.sh

Raw evidence stays under ignored results/ until the full count, source
commit, server flags and per-case metrics pass verification. Record
negative cases and failures; publish to GitHub as the only source of
truth after verification.

## Next experiment after this gate

Design a deadline-based admission controller with explicit accepted,
rejected and timeout counts. Compare baseline and control under
**exact same offered** inputs/schedules and keep SLO denominator all
offered requests. Dropping 50% to make accepted P95 fast is not a valid
unqualified service improvement.
