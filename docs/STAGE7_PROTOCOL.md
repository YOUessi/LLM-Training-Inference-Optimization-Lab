# Stage 7 — Frozen multi-genre natural-prose serving experiment

**Preregistered 2026-10-28 before Tang GPU measurements.**

## Research question

Do the simple Stage 5 conclusions about the P95 time-to-first-token
(TTFT)/throughput frontier persist when both prompts and generated output
lengths vary naturally? Detect and quantify any meaningful differences,
rather than assuming earlier synthetic benchmarks generalize.

## Dataset (fully pinned, public-domain in USA)

- 30 full-sentence, nonoverlapping excerpts from five public-domain prose works.
- Social fiction, science fiction, natural science, travel fiction and fantasy.
- Corpus SHA-256:
  `2912a4d8cc17789a1b6b339cac927a27ec41f24f7de140d84466ecd441e095ca`.
- Frozen excerpt text, sources and copyright cautions:
  `data/stage7_public_domain_passages.jsonl`,
  `data/stage7_public_domain_manifest.json`, `data/README.md`.
- Each 24-request case uses 24 distinct passages, exactly 6 per input
  length, with all five genres represented. Same seed yields exactly the
  same content/order across concurrency and short/long response cases.
  Different seed (42/137) changes selected documents and their order.

## Real tokenizer calibration

On Tang, Qwen2.5-1.5B-Instruct tokenizer loaded offline using frozen passages.
Char-cutoff calibrations (before final prompt wrapper) were:

| Input bucket | Frozen excerpt chars | Approximate model prompt tokens | Allowed actual server tokens |
|---|---:|---:|---:|
| short | 550 | 118–190 | 110–240 |
| medium | 1700 | 356–514 | 345–580 |
| long | 3900 | 778–1041 | 740–1140 |
| very_long | 6000 | 1165–1593 | 1120–1720 |

**Server-reported `prompt_tokens` must satisfy the preregistered bound
for each individual request**. Settings like `max_model_len=2048` and
char counts are not treated as actual input length measurements.

## Factorial serving test

- Real vLLM 0.19.1, Qwen2.5-1.5B-Instruct FP16, Tang RTX 4090 Laptop.
- Same model snapshot as Stage 3–6. CUDA 12.8 Torch 2.10.0.
- Prefix caching **explicitly OFF** throughout, no GPU-block override.
- Local loopback 127.0.0.1:8026, max model context 2048,
  max-num-seqs 8, GPU memory utilization 0.50.
- Seed 42 runs concurrency 1→2→4→8, seed 137 reverses 8→4→2→1.
- Two generation modes:
  - short: max_new_tokens=32, one-sentence summary request,
    mean **actual completion_tokens** must be >=8;
  - long: max_new_tokens=128, detailed 20-observation request,
    mean actual completion_tokens must be >=85.
- 24 measured requests per case, 1 excluded warmup request per case:
  **4 concurrencies × 2 seeds × 2 modes × 24 = 384 measured requests**.
- All 24 must succeed. Provider prompt/output token usage must be complete.
- Save per-request TTFT, TPOT estimate, total latency, output tokens,
  book/document ID, genre, excerpt chars and SHA-256; no generated text logged.
- Per-length P50/P95 TTFT and overall output tokens/s and request/s.
- Synthetic excerpt task is fixed, but genres and unmodified book prose vary.
  Do not claim general production representativeness.

## Acceptance and interpretation

Do not promote any SLO recommendation if input lengths fail, generation
is unexpectedly short, service errors occur or corpus SHA changes.

For each concurrency and output mode, conservatively aggregate across
two seeds: **max observed P95 TTFT, min measured output throughput**.
Compare long- vs short-output curves and Stage 5 only with explicit warnings:
the corpus and decode budgets changed, so this is a distribution-shift test,
not a controlled single-parameter optimization attribution.

P95 estimators from only 24 requests per case and 6 per length bucket are
descriptive and are NOT statistically strong tail latency guarantees.

## Execution

Run on Tang with idle GPU and port 8026 free:

```bash
bash scripts/run_stage7_natural.sh
```

Raw JSON is under ignored `results/stage7_natural_20261028`. The experiment
launcher owns and cleans up only the vLLM process it creates.
The raw model text is not stored in results: passages are pinned in the repo.
The complete raw results, failure modes and SHA256 should then be archived
in GitHub only **after** all gates pass.
