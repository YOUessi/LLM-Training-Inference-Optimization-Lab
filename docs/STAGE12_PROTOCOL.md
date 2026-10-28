# Stage12 — Cross-domain admission-policy validation (approved finite jobs)

Preregistered 2026-10-28 before Tang GPU execution.

## Research question

Stage11 reproduced client-side bounded admission SLO-versus-rejection
tradeoffs using two unseen arrival seeds, but still reused the same
Kubernetes documentation topics. We now change the source-content domain
without tuning the old 150/400ms admission policies.

## Frozen independent-domain workload

- 24 previously source-verified Project Gutenberg literary/natural-science
  passages from five different public-domain-in-US classic books.
- English only; this is NOT an independent multilingual fairness test.
- Exactly six distinct requests per Qwen-token-calibrated input length:
  170 / 300 / 470 / 680 Tokens. Provider-reported real prompt Tokens
  must satisfy the frozen ±95 acceptance window.
- Frozen request SHA256:
  fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04
- Exact Qwen2.5-1.5B-Instruct tokenizer snapshot
  989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- Original author, source eBook, excerpt hashes and source license scope:
  data/STAGE12_SOURCE_RIGHTS.md and Stage7 Project Gutenberg manifest.
- This is a distribution shift from the technical-document baseline,
  not a representative production, real-time or private enterprise corpus.

## Controlled model/serving setup

- Tang NVIDIA RTX4090 Laptop, one GPU, vLLM 0.19.1,
  Qwen2.5-1.5B-Instruct FP16.
- max-num-seqs=8, max-model-len=2048, cache OFF, 127.0.0.1:8026.
- 32 client executor workers; all policies get identical data and
  fixed-duration randomized arrival schedule, average offered 8 req/s.
  This is not an unconditioned homogeneous Poisson process.
- Held-out seeds **4093** and **5087**, different from earlier
  42/137 and Stage11 2027/3031.
- Two output modes (brief label and 128-token-budget detailed answer).
- Three policy configurations:
  (1) unlimited baseline; (2) 8 in-flight/4 queue slots + 150ms admission
  deadline; (3) same with 400ms deadline.
- Two seeds × 2 modes × 3 policies × 24 offered = 288 real offered requests.

## Acceptance, all-offered denominators, negative controls

Every scheduled offer is recorded as completed, rejected or failed. Reject
reasons are explicit, and every reject is a failure of timely-first-token
SLO. Report SLO compliance at 100/250/500/1000ms over **ALL** 24 offered
requests in each case, accepted/rejected counts, useful output Tokens/s,
actual provider prompt/completion Tokens, observed client & server queues,
plus per-original-book or topic rates.

The existing Stage10 client-side semaphore policy is unchanged; its
control values are just frozen as above. No *new* vLLM internal scheduler
or adaptive admission model is being claimed.

The same 24 prompts and seeded arrival schedule must be used across
the three policies. A fall in rejection count at 400ms is not by itself
a global gain if SLO compliance and useful throughput deteriorate.
All negative/regressive outcomes are retained and documented.

## Actual 10-minute automation

GitHub Actions every 10 minutes performs a read-only CPU preflight on
the frozen Stage12 corpus and runs full source/evidence tests.
An authorized Tang user-cron executor, also every 10 minutes, advances
a finite SHA-pinned allowlist:
  Stage12 seed 4093 -> Stage12 seed 5087 -> CPU cross-domain analysis.

The runner refuses a changed Git HEAD, dirty worktree, occupied GPU,
busy port or concurrent experiment, and stops on any failure.
There is no automatic git pull, package installation, uncontrolled
code-writing agent or ChatGPT background conversation.

Stage12 jobs are preapproved in the reviewed repository version only;
after these three jobs finish, the queue becomes status-only. New GPU
code requires a new reviewed source pin.

## Limitations

This is 24 offers per case and only two independent seeds. The original
five books are English works dated well before modern interactive
workloads. Therefore an observed difference does not prove production
generalization, fair multilingual admission, or superior LLM scheduling.
A true adaptive admission policy requires a separate engineered algorithm
and held-out comparison that reports rejection cost.
