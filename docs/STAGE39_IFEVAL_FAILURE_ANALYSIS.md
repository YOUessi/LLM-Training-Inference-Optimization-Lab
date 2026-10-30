# Stage39: why Dolly SFT damaged independent instruction constraints (diagnosis, NOT causality)

Verified Stage36 original **Google IFEval96 frozen** result with original official constraint checker, **all 96 paired prompts**, original Qwen3-8B and two actually trained NF4 adapters. This is an examination of a **previously observed development dataset** and cannot be reused as an independent test when changing adapter strengths in Stage38. Actual generated answers are archived under `evidence/stage36_official_ifeval/`. The reproducible source uses `analysis/stage39_ifeval_failure_audit.py` and machine-result `evidence/stage39_diagnostics/analysis.json`.

| Measured / descriptive diagnostic | Original 8B | Short SFT | Full 512-record SFT |
|---|---:|---:|---:|
| Strict IFEval prompts passed /96 | **71** | 43 | 52 |
| Responses reaching 384 generated tokens /96 | 31 | 36 | 42 |
| Strict pass rate among **not capped** outputs (biased strata) | 84.6% | 56.7% | 74.1% |
| Word bigram diversity (distinct-2, heuristic) | 0.853 | 0.452 | 0.421 |
| Low-distinct-2 / strongly repeating outputs (threshold0.25; **heuristic**) | **1/96** | **35/96** | **38/96** |

**Matched rule-correctness discordances:** Original base correct and full-SFT wrong on **24** examples; full SFT correct and original wrong on only **5**. Exact paired McNemar two-sided p≈0.00055 for this one fixed subset; base vs short 32 base-only and 4 short-only (p≈1.94e-6). These are evidence of **a strong conditional regression**, not proof which training mechanism caused it. Bigram diversity is tokenization/language dependent, particularly Chinese and creative writing; no universal "hallucination" label follows from this heuristic. Model-dependent "not capped" subsets cannot be treated as causal controls.

**Implication for project goals:** Stage31's Dolly assistant CE improvement (from 2.815 to 1.614 on its heldout reference) did **not** preserve general-purpose constraint obedience. The prior training objective optimizes reference next-token likelihood rather than general instruction coverage. Longer training increased cap-hit and low-diversity responses, suggesting exposure bias/repetition or formatting collapse as plausible *hypotheses*, not proven mechanisms. Stage38 pre-registers five residual adapter strength settings using a completely different IFEval dev43 and **untouched 101-test** to search for a Pareto-safe inference intervention with Dolly CE constraints. If no adapted setting clears dev constraints, select original base and report failure. Do not overclaim that reducing adapter strength constitutes a trained new alignment algorithm.
