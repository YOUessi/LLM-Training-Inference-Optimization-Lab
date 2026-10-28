# Stage10: real client-side bounded admission and all-offered-request SLO

**Frozen 2026-10-28 before real GPU execution.**

## Problem / hypothesis

Under 8 req/s and detailed 128-token decoding, Stage9 showed the load
generator bottleneck migrates from its client executor (8 threads) to
vLLM waiting requests (32 threads) without materially lowering overall
scheduled-arrival P95 TTFT.

A bounded *client-side* admission proxy may reduce latency for requests
it accepts, but **must be evaluated across every offered request** and
cannot be called an overall optimization if it only rejects/loses traffic.

This is not a modification of vLLM internals, and it does not represent
production deployment.

## Controlled A/B policies

Server: vLLM 0.19.1, local Qwen2.5-1.5B-Instruct FP16 on Tang RTX4090
Laptop GPU; 127.0.0.1:8026, context length 2048, max-num-seqs 8;
prefix caching OFF; GPU memory target 0.50.

Both policies use a fixed-capacity **32-worker client executor** so that
load-generator thread queuing is not the intended lever.

Policy A: unlimited direct-submit to vLLM; no local admission rejection.
Policy B: at most 8 in-flight requests, at most 4 client-side waiting
slots; each request must acquire an in-flight slot within 250 ms of
its scheduled arrival, or it is **explicitly rejected**. If the waiting
queue is full, reject immediately. Semaphore wake ordering is NOT
strict FIFO; no claim of fairness guarantee.

Same exact 24 frozen multilingual technical input prompts and seeded
fixed-duration randomized arrival schedules in each case; target and
realized average offered rates are 8 req/s. Same seed produces the
same arrivals, request identities and model output instruction
under both A and B.

Conditions: 2 output modes (label, detailed) × 2 policies × 2 seeds
(42/137) × 24 offered = **192 offered requests**. Seed 137
reverses treatment and mode execution order to reduce simple drift.
One excluded model warmup per case.

## Required truthful measurements

A request status is one of: ok, error, rejected. Rejections have
an explicit reason, queue_full or deadline_expired. Silent dropping,
unknown status and unaccounted requests are INVALID.

- offered=24, accepted+rejected=24, completed+errors=accepted.
- Real model prompt_tokens/completion_tokens required for ALL completed.
- Rejected and errored requests are *automatically NOT SLO-compliant*.
- All-offered TTFT SLO compliance at 100/250/500/1000 ms is
  compliant completions divided by **24 offered**, not only completed.
- Report accepted-only P95 arrival-to-first-token separately.
- When any offer is rejected (or errors), all-offered P95 latency
  cannot be observed as a finite complete value: record it NULL.
- Report useful model output Tokens/s, accepted %, rejected %,
  error %, per-language completion/rejection count, client admission
  wait, and sampled actual vLLM waiting/running and KV usage gauges.
- No policy superiority should be claimed from accepted-only P95 or
  successful throughput without all-offered compliance and drops.

## Acceptance

Every offered request has exactly one final record. Any accepted result
must have a correct SSE/model completion and pass actual provider input
and output length checks. Every dropped/expired offer is counted as
rejected; expected rejection does not cause the benchmark to discard
the whole experimental case.

Missing source identity, request record, actual GPU service, counters
or unknown error causes INVALID result. No prior trial may be overwritten.

Run on Tang with idle GPU:

    bash scripts/run_stage10_admission.sh

Raw files under ignored results/ are archived to GitHub only after
the experiment, per-request accounting and actual-source checks pass.

## Limits

One RTX4090 GPU, fixed Qwen model, Kubernetes documentation domain and
two arrival seeds with 24 requests per condition. Results are
exploratory and not a high-confidence service-level guarantee.
Client admission is a simulated proxy, not a vLLM engine scheduler
change. Do not claim improvements in a proprietary policy or model.

## Post-run observation (append-only, does not change acceptance thresholds)

The matched real GPU study completed **192/192 offered requests accounted for**:
186 completed and 6 explicitly rejected, zero unexpected errors. The
unlimited baseline completed all requests. Bounded detailed policy rejected
3/24 offered under each seed and improved 250ms first-token all-offered
compliance to 87.5% versus 75.0% and 66.7%, but reduced useful output
throughput. Label-mode negative controls rejected zero traffic under either
policy. See STAGE10_REPORT_2026-10-28.md and Stage10 raw SHA256 manifest.
Do not present this as a universally beneficial server-side optimization.
