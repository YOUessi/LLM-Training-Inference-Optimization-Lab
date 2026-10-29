# Stage35 OASST1 real human-request source vLLM cached-prefix comparison

**Measured**: 24 real Qwen3-8B-AWQ GPU cases ×96 original community user prompts = **2,304 offered/completed**, three arrival seeds, original-source Apache-2.0 licensed OASST1 root questions in English/Spanish/Chinese, 32 per language. Original user text is retained only locally for privacy; GitHub contains public source SHA, deterministic filtering recipe, anonymous source hashes and complete numeric per-request GPU evidence (no raw user text). Exact paired question IDs, system prefix state, arrival time and prompt SHA verified; vLLM prefix cache ON/OFF under same condition. Server restarts per arm, output capped at **128 tokens**.

| Fixed real-user prompt treatment | Offered rate | Mean useful output TPS ON vs OFF | Mean all-offered SLO500 change | Mean P95 TTFT change | Output cap hit-rate (ON) |
|---|---:|---:|---:|---:|---:|
| cold0 | 8/s | -0.02% | +0.00 pp | +0.6 ms | 96.2% |
| hot100 | 8/s | +18.78% | +72.57 pp | -2254.5 ms | 95.8% |
| hot50 | 4/s | +0.16% | +0.00 pp | -77.7 ms | 95.8% |
| hot50 | 8/s | +3.75% | +39.24 pp | -777.6 ms | 95.8% |

**Critical negative evidence:** A very high fraction of original OASST user requests hit the 128-token answer cap. Thus even a high 500ms first-token SLO / tokens-per-second does **not** establish that the assistant supplied a complete human-usable answer. The fixed cap intentionally keeps decode work bounded and permits controlled system comparison; Stage37 has independently frozen output384 conditions and actual vLLM KV-prefix hit token counters to evaluate the tradeoff. Do not claim an end-to-end SLA or answer quality until they are resolved.

Unlike Stage30 PSF docstrings, this is public human-contributed conversational content, but it is filtered and evenly sampled in 3 languages: **not** a representative production workload, and the 0/50/100%-common system instructions were synthetically introduced. Cache ON/OFF is matched within each prompt mixture, but comparing hot100 with cold0 directly changes model instructions and is not a pure caching treatment. Runtime includes a single unrelated warmup per fresh server, not a cold GPU boot. Only three independent arrival seeds per condition; all bootstrap CIs are exploratory.

Raw result files and checksums: `evidence/stage35_real_human/`, analysis source `analysis/stage35_human_analysis.py`, frozen source selection `data/stage35_oasst_manifest.json`. Original prompt text stays in ignored Tang `results/stage35_public_data/` to minimize unnecessary republication of possible personal data. No human preference correctness or claim of natural prefix-hit rate.
