# Stage11 — Held-out admission study and safe 10-minute execution

Frozen plan: 2026-10-28. This is an approved FINITE task queue, not an unrestricted agent.

## Scientific question

Stage10 measured a first-token SLO improvement at the cost of rejecting
6 of 48 detailed offered requests and reducing useful output throughput.
Validate admission independently with never-before-used arrival seeds.

- Model: local Qwen2.5-1.5B-Instruct FP16, one Tang RTX 4090 Laptop.
- vLLM 0.19.1, max-num-seqs=8, max context=2048, prefix caching OFF.
- Same 24 frozen Kubernetes CC BY 4.0 multilingual documents used
  in Stage10, so this is NOT an independent-domain validation.
- Fixed-window randomized arrival spacings: average 8 offered requests/sec,
  NOT an unconditioned homogeneous Poisson process.
- Held-out seeds 2027 and 3031.
- Two output modes: brief resource label and detailed analysis.
- Three policies: unlimited; bounded at 150 ms admission deadline;
  bounded at 400 ms deadline. Bounded means 8 in-flight and 4 queue slots.
- Total: 2 seeds x 2 modes x 3 policy arms x 24 = 288 offered requests.
  Reverse order between seeds; preserve every rejection and error.

Report 100/250/500/1000 ms first-Token SLO compliance divided by
EVERY offered request, not only those accepted; preserve useful
completion token/s, rejected counts, by-language fairness, actual
prompt/completion usage and client versus server queue measurements.

If any offers are rejected, all-offered P95 completion latency is
not observable and must remain null; do not substitute accepted-only P95.

## GitHub Actions every 10 minutes

Workflow: .github/workflows/plan-audit.yml
UTC minutes: 07,17,27,37,47,57.

Every scheduled execution runs CPU regression tests, a deterministic
preapproved Stage11 request/seed/experiment preflight, roadmap-vs-evidence
comparison and a JSON artifact. GitHub-hosted runners do NOT have access
to Tang GPU and cannot send progress messages to this ChatGPT chat.
GitHub schedule is best-effort and may be delayed or skipped.

## Tang local cron every 10 minutes

Local user cron runs scripts/auto_progress_tick.py on Tang at minutes
00,10,20,30,40,50, when the machine is on. It executes AT MOST ONE
preapproved job per invocation and has this static finite queue:

1. Held-out seed 2027: 144 real GPU offered requests
2. Held-out seed 3031: another 144 real offered requests
3. CPU independent analysis of all 288 offers

Safety: manually pin the exact full Git commit in the ignored file
results/auto_progress/approved_head.txt, refuse changed Git HEAD or
dirty worktree, use exclusive filesystem flock, and perform CPU tests.
The runner never does a git pull, package install or driver change.

GPU jobs are gated on nvidia-smi compute-process ownership and an unused
127.0.0.1:8026 port. They have a 600-second bounded timeout and only
terminate the vLLM instance they started. Failed/interrupted experiments
are NOT silently retried. All raw results and logs remain in ignored
results/auto_progress and results/stage11_seed_2027 or seed_3031.

Once the finite approved queue is empty, the cron checker continues
to report status but DOES NOT invent new code or GPU experiments.
Any further workload requires a reviewed source revision and a new pin.

Inspect:

    cat results/auto_progress/state.json
    tail -80 results/auto_progress/cron.log
    tail -80 results/auto_progress/heldout_seed_2027.log

To stop local execution, remove only the crontab entry marked
LLM-OPT-AUTO-PROGRESS. To stop the cloud checker, remove the scheduled
trigger from the workflow.

Local cron is a real on-machine scheduled process, not a ChatGPT task
automation. Results are not posted automatically into this conversation.
GitHub remains the fact source for code and reviewed public evidence;
unfinished local GPU logs are not yet published research results.

## Scientific limits

Only 24 offers per condition and two held-out arrival seeds; no confidence
intervals or production SLA guarantees. Same Kubernetes-domain corpus;
not a cross-domain generalization study. Client-side admission proxy,
not an internal vLLM scheduler optimization. Rejected requests are SLO
failures and must be reported without hiding the throughput cost.
