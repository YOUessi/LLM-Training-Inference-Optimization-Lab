# Stage 8 — Real multilingual cloud-native serving with open-loop arrivals

**Frozen on 2026-10-28 before running the GPU benchmark.**

## Goal and data provenance

Replace Stage7's five classic English literary works and fixed concurrent
bursts with contemporary **technical documentation in three languages** and
Poisson-like exogenous request arrival times.

- Upstream: the Kubernetes website documentation, CC BY 4.0.
- Exact upstream commit: `ed70e311737a341574ed5758b7af24e3bba232cb`.
- Attribution: Kubernetes Authors and contributing localizers. Changes:
  remove website YAML, shortcodes/code samples, headings, links and extract
  disjoint windows. See `data/STAGE8_SOURCE_LICENSE.md`.
- 24 distinct excerpts: 8 English, 8 Simplified Chinese, 8 Japanese;
  four cloud-native subtopics (Pods, Deployment, Secrets, Persistent Volumes).
- **Limits:** This is not a multi-industry/production dataset. Text excerpt
  windows may begin mid-sentence; no claim of fully natural production requests.
- Exact excerpt corpus SHA-256:
  `d59076782e8fe177eae2b2029bef0f7ce479b86f6fc65214fb60e8e3f734b5c4`.
- Frozen model-token-calibrated request set SHA-256:
  `c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c`.
- Qwen2.5-1.5B-Instruct offline tokenizer, exact snapshot
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- All three languages contribute exactly 2 docs to each of four
  actual calibrated input-length buckets, approximately
  170, 300, 470 and 680 tokens; each *server-reported* prompt token count
  must remain within ±95 of the target to permit extra output instructions.

## Real GPU runtime

- Tang RTX 4090 Laptop GPU, vLLM 0.19.1, PyTorch 2.10.0+cu128.
- Single isolated model service bound only to 127.0.0.1:8026.
- Automatic prefix caching explicitly OFF; no KV block override.
- Max model context 2048; GPU memory utilization target 0.50;
  max-num-seqs 8.
- A 24-request case contains 8 requests per language, 6 per length bucket,
  across the four topics. Seed 42/137 reshuffles arrival order but does not
  change the frozen prompt bodies or bucket assignments.

## Arrival and output experiment

One **open-loop** request scheduler uses deterministic seeded exponentially
distributed gaps (Poisson interarrival construction), independent of response
completions, at target offered rates **2 and 8 requests/second**.

All cases use the *same* maximum output cap of 128 Tokens. Three response
instructions ask for:

1. `label`: a very short noun phrase; at least 75% of actual responses
   should stop within 32 Tokens (natural early stopping).
2. `summary`: one or two sentences; at least 50% should stop before
   128 Tokens.
3. `detailed`: at least 20 technical points; mean actual generated Tokens
   must be at least 80 (long-decode load).

Matrix: 3 generation modes × 2 arrival rates × 2 seeds × 24 measured
requests = **288 real measured requests**, plus 12 excluded warmups.
Seed 42 uses low-to-high rate and label/summary/detailed order;
seed 137 reverses both orders. This reduces simple order bias but
does not guarantee statistical independence from server warmup/temperature.

## Precise metrics and acceptance gates

Each request stores scheduled arrival offset, actual client worker dispatch,
executor queue wait, TTFT from the actual network send, TTFT from its
*scheduled* arrival, total completion time, actual input Tokens,
actual output Tokens, source topic/language ID and prompt SHA-256.
No generated answer text or model auth headers are stored.

- P95 TTFT is computed both from actual dispatch and from scheduled
  arrival. The latter includes **client executor** queueing; it is not
  an isolated measurement of vLLM internal queueing.
- Offered interarrival rate is computed from first-to-last planned
  arrival; completed request throughput is measured separately.
- P95 arrival-to-first-token SLO violations are recorded for thresholds
  100/250/500/1000 ms, *not* preclaimed production SLOs.
- During each case, a 150 ms background poll of real vLLM Prometheus
  metrics samples KV Cache utilization **if the gauge exists**.
  Unavailable series are null, not interpreted as zero.
- 24/24 requests must succeed; provider prompt/completion token usage
  must both be 100%; all four bucket token checks must pass; the mode's
  *actual* output length gate must pass. Otherwise the case status is INVALID.
- Run only if no other GPU task is present and localhost port is unused.
  Stop only this script's own vLLM process after the study.
- Raw case JSON, Prometheus snapshots, frozen input/source hashes,
  actual server cache flag, and failures must be retained. Never silently
  widen a length gate or mix failed runs into accepted performance figures.

## Planned commands

Run only after code and request data are committed to GitHub, synced to
Tang's matching source commit and CPU tests passed:

```bash
bash scripts/run_stage8_arrivals.sh
```

The two-seed matrix is a reproducible research harness, **not a replay
of production user traffic**. Do not attribute performance differences
to PagedAttention/KV architecture or a trained scheduling policy.

## Limitations and next scientific step

24 requests per setting and only two seeded schedules cannot give
high-confidence P95 latency guarantees. Real inference quality is
not scored. Later experiments should include independent non-Kubernetes
domains, natural burst processes from actual anonymized workloads,
longer outputs and a candidate admission/queuing policy evaluated
against an untouched held-out request set.
