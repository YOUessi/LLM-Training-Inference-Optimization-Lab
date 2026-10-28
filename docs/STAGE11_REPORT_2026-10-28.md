# Stage 11 — Held-out Admission Policy Validation with 10-Minute Automated Execution

**2026-10-28. Result: 288 actually offered requests, 278 completed, 10 explicitly rejected, zero unexpected errors.** Two held-out arrival seeds ran on Tang RTX 4090 using a version-pinned, guarded user-cron-compatible executor. This is independent *arrival-seed* validation, not a different data domain.

## Hypothesis and frozen controls

- Stage10 found a 250ms first-token SLO improvement accompanied by 12.5% rejection and lower useful throughput. We kept those original seeds (42 and 137) out of this validation and used new held-out seeds 2027 and 3031.
- GPU: NVIDIA RTX 4090 Laptop, model Qwen2.5-1.5B-Instruct FP16, vLLM 0.19.1, max-num-seqs 8, model context 2048, automatic prefix caching OFF.
- Dataset: the same 24 pinned Kubernetes CC BY 4.0 technical prompts in English, Simplified Chinese and Japanese. Real prompt and completion Token gates passed for every completed request.
- Workload: conditioned random interarrivals at a matched average **8 requests/second** (not an unconditioned homogeneous Poisson process). 24 offers per experiment, 2 seeds × 2 output modes × 3 policies = 12 actual experiments, 288 offers total.
- Policies: unlimited baseline, bounded 8 inflight + 4 waiting slots and **150ms** deadline, same bounded capacity with **400ms** deadline. The nominal baseline 250ms deadline argument is unused for baseline.
- Every rejection is counted as noncompliant. SLO250 numerator: offers that actually emitted the first Token <=250ms after scheduled arrival. Denominator: ALL offered requests, not only accepted completions.
- Experiment source Git commit: `e5356b799e9916fbda91f79ba99703c08f007b62`. Raw run files, independent analysis and local automation status are tied to that immutable source.

## Combined results across two held-out seeds (48 offers per mode/policy)

| Output mode | Admission setting | Completed / offered | Explicit rejected | All-offered first-token SLO<=250ms | Minimum observed useful output tok/s |
|---|---|---:|---:|---:|---:|
| label | No admission | 48/48 | 0 | 100.0% (48/48) | 24.1 |
| label | Bounded, deadline 150ms | 48/48 | 0 | 100.0% (48/48) | 24.1 |
| label | Bounded, deadline 400ms | 48/48 | 0 | 100.0% (48/48) | 24.2 |
| detailed | No admission | 48/48 | 0 | 72.9% (35/48) | 734.4 |
| detailed | Bounded, deadline 150ms | 42/48 | 6 | 87.5% (42/48) | 694.6 |
| detailed | Bounded, deadline 400ms | 44/48 | 4 | 81.2% (39/48) | 699.7 |

## What was actually learned

- For detailed output the unlimited baseline scored **35/48 (72.9%)** timely first tokens with no rejection.
- With the 150ms admission deadline it scored **42/48 (87.5%)**, but refused 6/48 detailed offers (12.5%). This is a **14.6 percentage-point improvement** under the all-offered denominator, obtained at a service coverage cost.
- With the 400ms deadline it scored **39/48 (81.2%)**, rejected 4/48 (8.3%), and maintained fewer timely first tokens than the 150ms case.
- Label-mode negative controls completed all 48/48 with **no rejection** and 100% timely first tokens under all three policies.
- The 150ms candidate rejected **2 English, 2 Chinese, 2 Japanese** offers in aggregate; the 400ms candidate rejected **1 English, 0 Chinese, 3 Japanese**. Only 16 offers per language across two seeds under each detailed policy, so the apparent disparity is too small to claim systematic language unfairness; nevertheless it is explicitly reported and should be audited with larger traces.

**Scientific judgment:** Independent arrival seeds reproduce the workload-specific latency-versus-rejection tradeoff. They do not establish a universally better policy. No vLLM internal scheduler was modified; this is a client-side admission-proxy baseline.

## Actual automatic execution and safeguards

- GitHub Actions was changed from a 30-minute to a **10-minute best-effort** schedule (UTC 07/17/27/37/47/57). It runs CPU/evidence tests, preapproved Stage11 input/seed checks and a roadmap audit.
- Tang's running /usr/sbin/cron service has a user-level 10-minute check (minutes 00/10/20/30/40/50). Its executor requires a manually pinned full Git commit and clean worktree; does not pull mutable GitHub main or change global packages.
- A file lock prevents overlapping ticks. Before each GPU task, NVIDIA compute-process and localhost port checks refuse resource conflicts; a timeout bounds the experiment and its owned service is cleaned up.
- The executor ran three allowlisted tasks successfully: heldout_seed_2027, heldout_seed_3031, then independent CPU analysis. Their final state and log hashes are persisted, and the queue is now **exhausted**. No new unapproved jobs are generated.
- New GitHub commits require an explicit review of the SHA pin and state migration before more GPU tasks can run. The local scheduler does not send updates into ChatGPT.

## Original evidence and reproducibility

- [All 12 real SSE raw cases](evidence/stage11_heldout/), including each rejected offer's reason, per-language request outcome, completed provider Token usage and client/server queue metrics.
- [Source provenance + complete per-file SHA256 manifest](evidence/stage11_heldout/stage11_manifest.json).
- [Independent machine-generated 288-offer / 12-case analysis](evidence/stage11_heldout/stage11_analysis.json).
- [Final user-cron automation job states](evidence/stage11_heldout/auto_progress_state.json).
- [Preregistered research and 10-minute automation safety protocol](STAGE11_AUTOMATION_PROTOCOL.md).
- [Stage10 earlier tuning sample with seed 42/137](STAGE10_REPORT_2026-10-28.md).

## Remaining limitations

1. This is **arrival-seed holdout only**; source content remains Kubernetes technical documents. No independent financial, medical or nontechnical dataset was tested.
2. Only 24 offers per experiment and two held-out seeds: do not imply narrow confidence intervals or guaranteed production P95 latency.
3. Successful label mode does not demonstrate broad fairness: the detailed bounded modes reject requests and the per-language numbers are small.
4. The admission controller is a simple client semaphore with a deadline, not a new vLLM internal scheduling algorithm, and rejected requests still represent real failed service.
5. More deadlines, different request rates, longer traces, independent hardware and objective user costs are needed before recommending a policy for production.

## Next research gate

Freeze an independent-domain held-out corpus and multiple offered rates; compare a more principled queue-priority/admission strategy against the fixed 150ms/400ms baselines using all-offered timely completion, useful throughput, exact error/rejection accounting and per-language fairness. Only promote a policy if it improves a predeclared tradeoff without hiding losses.
