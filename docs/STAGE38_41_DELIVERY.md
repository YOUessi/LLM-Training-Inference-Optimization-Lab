# Stage38–41: real 8B training-instruction Pareto follow-up — final research audit

**Evidence audit PASSED for each distinct frozen input, dev selection, original adapter SHA and all completed model comparisons.** Model strengths and six constraint-rehearsal training models were selected only on separate dev sets; final IFEval source keys and new Dolly128 labels were never used for selecting the winner. **Production release remains BLOCKED** until independently verified real-user correctness and robust across-seed evaluation, regardless of any favorable per-task result.

## Stage38: inference-only LoRA strength and independent 101 IFEval test

| IFEval model | Strict obedience /101 | Strict rate |
|---|---:|---:|
| Original Qwen3-8B (α0) | 78/101 | 77.23% |
| Stage31 full rank16 (α1) | 52/101 | 51.49% |
| Frozen dev-selected α=0.75 | 67/101 | 66.34% |

Dev43 adapted α feasible: True. Independent test selected-minus-base bootstrap95 [-20.79,-1.98] pp. Stage41 totally unused Dolly128 answer CE: base 2.9784, selected strength 1.7514. Stage38 research-candidate strict go/no-go: **BLOCKED**. The word-level repeat diagnostic from Stage39 is supplementary and must not be retroactively inserted into the dev-selection policy.

## Stage40: six distinct real 8B deterministic constraint-rehearsal SFT runs + independent 60 IFEval test

| Model | Strict obedience /60 | Strict rate |
|---|---:|---:|
| Base | 47/60 | 78.33% |
| Previous Stage31 full naive SFT | 38/60 | 63.33% |
| Dev-selected Stage40 `base` | 47/60 | 78.33% |

Dev25 trained adaptation feasible: False. Independent test selected-minus-base bootstrap95 [+0.00,+0.00] pp. New Dolly128 answer CE: base 2.9784, selected constraint SFT 2.9784. Stage40 research-candidate strict go/no-go: **BLOCKED**. All 6 source conditions used the same original 512 Dolly identities twice, with source-conditioned JSON/XML/markers or no reformatting; results are not human preference labels.

## Stage41: 128 entirely new per-source identities, same Dolly upstream distribution
| Frozen model | 128 original human-reference answer CE | Perplexity |
|---|---:|---:|
| base | 2.97840 | 19.656 |
| original_full | 1.74027 | 5.699 |
| guarded_strength | 1.75139 | 5.763 |
| format_replay | 2.97840 | 19.656 |

## Go/no-go and honest limits

Even when official prompt-level IFEval on *new source identities* improves, a small hand-filtered public source cannot prove general instruction adherence or correct human answers. The Stage38 dev43/test101 and Stage40 dev25/test60 source keys are disjoint from each other and from the 96 previously observed Stage36 keys, but all are from the same original public IFEval collection; the test may still overlap original Qwen3 pretraining. Stage41’s 128 new question identities are disjoint from 832 previous Dolly selections, but again from the same public dataset and possibly pretrained. 95% bootstrap intervals resample fixed test cases within small frozen tiers, not independent training seeds. No decisions from the final tests were fed back into the score selection. The presence of an adapted-model winner is reported by explicit guardrails, NOT automatically implied by lower Dolly CE.

**Production deployment: BLOCKED.** Need real production load SLO + correct uncapped answers, human independent instruction/factual correctness, multi-dataset/seed generalization, and eventual multi-GPU training validation on multi-GPU hardware. GitHub code and raw stats remain the one source of truth, while nonselected large adapters are retained locally with SHA if necessary.

Full raw evidence: `docs/STAGE38_MEASURED_RESULTS.md`, `docs/STAGE40_MEASURED_RESULTS.md`, `docs/STAGE41_MEASURED_RESULTS.md`, `evidence/stage38_guarded_alpha`, `evidence/stage40_constraint_rehearsal`, `evidence/stage41_new_dolly128` and original IFEval Stage36 published negative result.
