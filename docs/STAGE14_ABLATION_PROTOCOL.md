# Stage 14 — Scheduler component isolation (protocol draft; NO GPU result)

**Status: CPU implementation and tests only.** Stage13's measured source and data are frozen; do not rewrite them or imply that the following variants have been benchmarked.

## Hypothesis

Stage13's one-of-eight preregistered candidate win may arise from adaptive admission, shortest-job-first, age compensation, stochastic burst composition, or a trade-off from rejecting requests. Separate each mechanism before claiming performance. The controller is a client-side proxy, not a vLLM internal scheduling patch.

## Factor isolation

Six CPU-defined variants: `fifo_static400`, `sjf_static400`, `sjf_aging_static400`, `fifo_dynamic`, `sjf_dynamic`, and `full_adaptive`. Pairwise contrasts isolate SJF under a fixed deadline, aging under fixed deadline, deadline adaptation under FIFO, deadline adaptation under SJF, and aging with dynamic deadline. Their EWMA enablement remains a possible confound for the `fifo_static400` reference; for a clean EWMA effect, add a separate no-EWMA counterpart before any GPU effect attribution.

## Execution gate (not yet authorized)

1. Review and freeze a fully integrated Stage14 driver and new non-overlapping document identities and arrival seeds. Existing Stage13 seeds are development evidence, not Stage14 independent validation.
2. Run all six policies on the identical offered request IDs and scheduled arrival offsets per matched cell. Randomize or counterbalance policy order. Use 8 and 16 offered requests/s and add a longer sustained high-load trace to stress queue behavior.
3. Use all-offered TTFT<=250 ms, rejected counts, useful output tokens/s, per-output-mode rejection, server running/waiting gauges, plus p95 and paired confidence intervals. Rejected offers are SLO failures. Apply same utility and throughput/rejection guardrails as Stage13. Do not substitute accepted-only latency.
4. Run on a verified idle Tang GPU only after CPU test clean pass and no concurrent GPU use. Preserve exact version, driver parameters, all raw results, failure logs and SHA256 checksums. An unapproved/unmeasured run must not be reported as complete.

Current implementation `src/llm_opt_lab/serving/stage14_ablation.py` isolates deadline/priority decisions, **not yet the full live request queue or vLLM integration**. The CPU tests verify functions, not real performance.
