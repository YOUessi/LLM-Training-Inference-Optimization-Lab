# Stage13 — Workload-aware, age-fair admission: frozen protocol

**Preregistered before Stage13 GPU experiments.** This is a **client-side
proxy policy, not a modification of vLLM engine internals or training of a
new model.**

## Original research problem

Stage11 technical-document serving favored bounded 150ms admission whereas
Stage12 English natural prose favored bounded 400ms. This limited evidence
does not justify either static deadline as a universal optimum.

Stage13 implements a new adaptive request-level decision mechanism:
- Estimate predicted service demand based on output mode (label/detailed)
  and prompt tokens, using previous Stage10 complete-response medians:
  label 55.49ms, detailed 1077.63ms.
  Prediction = prior × (0.84 + 0.16 × input_tokens / 390).
- Online per-mode EWMA (alpha 0.20) updates from actual completed requests.
  No future completion token counts are leaked to the estimate.
- Enforce at most 8 local in-flight and 4 waiting requests.
- When capacity frees, prioritize lowest predicted service milliseconds
  minus 2 × waiting-time milliseconds (short-job-first with aging).
  This is not guaranteed strict FIFO or provably starvation-free.
- Dynamic wait deadline in milliseconds, clipped to [100,400]:
  135 + 0.20 × min(predicted_service_ms, 1200)
      - 110 × active_requests / 8
      - 65 × waiting_requests / 4
      + 0.02 × (input_tokens - 390).
- Report explicit queue_full and deadline_expired rejection outcomes.
  Rejected work always counts against all-offered service compliance.

**Comparison policies:** unlimited direct submit, bounded FIFO fixed150,
bounded FIFO fixed400, and the age-fair adaptive policy.
Adaptive comprises priority plus changing wait budget; the two effects
have NOT been individually ablated.

## Real data/hardware

- Kubernetes technical-document corpus (24 frozen requests across English,
  Simplified Chinese and Japanese): SHA256
  c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c
- English classical prose / natural-science corpus (24 frozen prompts
  across five genres): SHA256
  fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04
- BOTH source domains were used before; Stage13 has **new arrival seeds**
  but NOT unseen source domains. Do not call it source-domain holdout.
- Qwen2.5-1.5B-Instruct FP16 on Tang RTX4090 Laptop GPU, vLLM 0.19.1,
  max model context 2048, max-num-seqs 8, GPU memory cap 0.5,
  prefix caching OFF, loopback-only 127.0.0.1:8026.
- Per case, seed-shuffle 24 distinct input documents and alternate
  12 short-label with 12 detailed-128-token tasks, creating a genuine
  mixed short/long decoding workload.
- 32 client executor workers to reduce self-generated client queue.
- Each paired policy must receive identical requests, output task
  assignments and timed arrivals.

Predeclared validation grid:
- source domains: k8s, prose
- seeds: k8s 6101 and 6109; prose 7201 and 7209
- offered rates: 8 and 16 per second, each with a seeded randomized
  fixed-duration schedule (NOT an unconditioned Poisson process)
- policies: unlimited, fixed150, fixed400, adaptive
- 2 × 2 × 2 × 4 × 24 = **768 real offered requests**, 32 measured cases
  plus 32 excluded warmups.
- Fresh vLLM service process for each domain/seed, reverse policy and
  rate order between seeds. No concurrent GPU jobs.
- All results retained under ignored local results until SHA-verified
  and committed to GitHub; negative results must be preserved.

## Frozen primary utility and acceptance

For each same-domain/seed/rate policy pair:
- S: fraction of ALL offered requests that emitted first Token <=250ms
  from scheduled arrival (rejected / errored are non-compliant).
- R: explicit rejects divided by ALL offered.
- T: real completed output Tokens/s divided by the paired unlimited
  baseline's output Tokens/s. No cross-domain normalization.
- Utility U = S - 0.50 × R + 0.20 × T.

These coefficients must not change after seeing results.

An adaptive policy is a *candidate win* for a matched scenario only if
U exceeds unlimited by >=0.02, measured useful throughput is >=80% of
unlimited and reject fraction is <=20%. Report also comparison with the
best static deadline arm on the same offered workload.
No candidate is promoted to cross-domain success without consistent
results across BOTH source domains and all four seeds.

Each 24-offer case must have exactly 12 label and 12 detailed tasks,
complete per-request source/timestamp evidence, total accounted outcome
24, actual model token usage and completion length gates when testable,
no unexplained errors, live queue/KV gauge observations, source SHA,
zero unauthorized GPU interference and all original negative results.

When any request is rejected, all-offered finite P95 response delay is
unobservable. The report must use all-offered 250ms compliance, explicit
rejection cost, useful output throughput and per-type/language coverage;
never quietly substitute accepted-only P95 as a guaranteed SLO.

## Limits and next test

The experiment is a single GPU, small model, just two source domains
already previously used and 24 offers/condition. vLLM internal scheduling
was not modified. Age fairness and online service demand are engineering
heuristics, not theoretical guarantees. Results do not establish a
production tail-latency confidence interval. A future study must ablate
priority versus deadline and use new source data and longer traces.
