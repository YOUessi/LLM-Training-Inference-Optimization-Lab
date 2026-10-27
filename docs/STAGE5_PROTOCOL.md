# Stage 5 — KV Cache capacity and mixed-length request experiments

Frozen initial protocol: 2026-10-28 (before GPU run).

## Goal 1 — demonstrate cache retention and eviction by *observed* counters

Start an isolated vLLM 0.19.1 Qwen2.5-1.5B FP16 server:
- --enable-prefix-caching
- --num-gpu-blocks-override 256 (bounded GPU KV block experiment)
- max_model_len 2048, max_num_seqs 8, gpu_memory_utilization 0.50
- host 127.0.0.1, port 8026; never public

Stage sequence, fixed seed/model/prompt:
1. initial cold request for 35× synthetic long prefix;
2. identical immediate replay: should hit >= 75% of queried tokens;
3. 12 distinct-prefix long requests to create KV cache capacity pressure;
4. identical anchor replay after capacity pressure: expect <=25% hits;
5. immediate repeat after the cold/evicted replay: expect >=75% hits.

Every observed cache hit fraction comes from differences of
vllm:prefix_cache_queries_total / vllm:prefix_cache_hits_total before/after
each request. A synthetic speedup alone is insufficient evidence.

If eviction is NOT observed, report the original result and adjust the
scientific hypothesis for the next experiment; do not silently mark PASS.
With 256 blocks, 16 tokens per cache block would imply about 4096 tokens,
but verify block size and the actual runtime settings from model logs.
This does NOT mean physical GPU memory was reduced by the same ratio
because model weights and graph allocations remain.

Run: bash scripts/run_stage5_eviction.sh

## Goal 2 — realistic mixed-context serving and latency/throughput tradeoff

Independently start normal-capacity vLLM with prefix caching explicitly OFF,
same model and CUDA device. Use four prompt-length buckets:
- short: 3× repeated technical context (approx. 140 prompt tokens)
- medium: 14× (approx. 500 tokens)
- long: 35× (approx. 1200 tokens)
- very-long: 47× (approx. 1600 tokens)

Server-reported prompt_tokens must lie within pre-registered bucket bounds
[85,250], [400,650], [1050,1400], [1450,1850] respectively.
Each 1/2/4/8-concurrency run has exactly 24 real requests
(6 per length), with two seeds 42/137 and measured token usage.
All lengths and all requests must pass; report P50/P95 TTFT and latency
per bucket, aggregate completion token/s and error rates.

Run: bash scripts/run_stage5_mixed.sh

## Auditability and safety

- Do not run if the NVIDIA GPU is held by another process or local port 8026 is in use.
- Every server process is started by a tracked runner and terminated at end of run.
- Real model text is neither printed nor included in the performance JSON.
- Raw JSON, errors, source commit, GPU identity and model version are archived.
- NoNsight GPU hardware counters remain permission-blocked.
- Stage5 work is considered complete only when results are verified and
  checked into GitHub with real evidence; mock tests are not GPU evidence.

## Recurring checks (GitHub Actions)

GitHub Actions workflow plan-audit.yml is scheduled using cron
'7,37 * * * *': every 30 minutes at minute 07 and 37 UTC (best effort,
platform may delay or skip). It runs repository CPU unit/integrity tests,
compares Roadmap and committed experiment reports, and publishes
an Actions run summary + JSON artifact, selecting the next verifiable
task. It CANNOT run Tang's GPU, change code autonomously, or post into
this existing ChatGPT conversation. Do not describe it as a continuously
working cloud-agent. ChatGPT Automations cannot schedule subhourly tasks.
