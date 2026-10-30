# Stage38: guarded adapter-strength recovery of official IFEval obedience

**Measured** real Qwen3-8B NF4 Stage31 rank16 LoRA. Strictly excluded **all 96 previously observed Stage36** IFEval keys. Preregistered **43 new development instructions** used to select inference-time adapter residual strength from 0/.25/.5/.75/1.0 based on BOTH Google official strict compliance and Dolly val96 answer-only CE/cap-hit guardrails. Final **101 different untouched IFEval source instructions** used exactly once after `frozen_dev_selection.json` exists. No parameter training or test-driven strength choice in Stage38. Output context384 greedy no-thinking. All source and adapter SHA checks, official grader per-case verdicts and actual generated outputs are archived in `evidence/stage38_guarded_alpha/`.

## Development-only Pareto preselection (not independent final result)

| LoRA residual strength | Official strict passed /43 | Dolly validation CE | Outputs at384/43 | All predeclared constraints passed |
|---|---:|---:|---:|---|
| 0.00 | 31/43 | 2.95435 | 12/43 | NO/baseline |
| 0.25 | 35/43 | 1.94652 | 13/43 | YES |
| 0.50 | 32/43 | 1.82293 | 15/43 | NO/baseline |
| 0.75 | 31/43 | 1.78725 | 13/43 | YES |
| 1.00 | 25/43 | 1.77701 | 19/43 | NO/baseline |

**Frozen selected α = 0.75**; genuinely adapted model satisfies all three dev guards: **True**. Guardrails were: dev strict success at least unadapted base−2, cap-hits≤base+2, and Dolly val answer CE reduction≥0.10. If none eligible, base α0 is automatically selected (no adapted win). No real IFEval final-test answer consulted by this selection.

## Final untouched IFEval101 official strict/loose test

| Fixed comparison model | Strict passed /101 | Strict prompt accuracy | Loose prompt accuracy | Cap384 hit fraction |
|---|---:|---:|---:|---:|
| α=0.00 | 78/101 | 77.23% | 78.22% | 25.7% |
| α=0.75 | 67/101 | 66.34% | 68.32% | 30.7% |
| α=1.00 | 52/101 | 51.49% | 54.46% | 33.7% |

### Predeclared selected minus comparators: within-fixed-test 95% stratified bootstrap interval (exploratory)

| Comparator | Selected−comparator strict accuracy interval |
|---|---:|
| base | [-20.79, -1.98] percentage points |
| full | [+6.93, +22.77] percentage points |

## Post-hoc repetition diagnostic (not an additional tuning rule)

| Scale | Low distinct-2 /101 (word bigram heuristic) | Mean distinct-2 |
|---|---:|---:|
| 0.00 | 0/101 | 0.866 |
| 1.00 | 29/101 | 0.497 |
| 0.75 | 23/101 | 0.562 |

**Judgment:** Higher Dolly reference CE performance alone is not an alignment result; preserve exact official IFEval prompt/constraint results, per-family failures, and 384-token caps. One set of 101 public IFEval source questions and one trained LoRA checkpoint do not establish production user safety, semantic correctness, or multi-seed robustness. Residual scaling is a legitimate inference-time Pareto control but not a newly trained model/architecture. Full source license Apache 2.0 and Dolly validation license CC BY-SA3.0. All raw development and test outputs are available in repo. The earlier 96 IFEval results informed the *hypothesis* and are not counted as independent verification.
