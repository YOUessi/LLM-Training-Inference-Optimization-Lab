# Stage40: real 8B validated constraint-format replay SFT vs original instruction obedience

**Actual measurements** of six Qwen3-8B NF4 QLoRA rank16 runs on one RTX4090 Laptop. Each run used 512 original Dolly CC BY-SA3.0 human-source identities twice (1,024 exposed source records and 512 optimizer updates), with fixed 0/25/50%-format augmentation and two independent train/data seeds 4011/4027. JSON/XML/begin-end constraint targets were algorithmically and deterministically transformed from human answers (not human-labeled new tasks). All training data and format verifiers had been frozen before these results. No IFEval prompt used as a training target. All six raw metrics and training logs archived in `evidence/stage40_constraint_rehearsal/`.

## Dev25: selection based on BOTH official instruction obedience and Dolly validation CE

| Candidate | Format share | Dolly96 validation CE | IFEval dev25 strict pass count | Output cap hits /25 | Dev guarded eligible |
|---|---:|---:|---:|---:|---|
| base | - | 2.9525 | 22/25 | 4/25 | NO/reference |
| old_full | - | 1.7770 | 14/25 | 8/25 | NO/reference |
| mix00_seed4011 | 0.0 | 1.7811 | 12/25 | 7/25 | NO/reference |
| mix25_seed4011 | 0.25 | 1.7809 | 16/25 | 10/25 | NO/reference |
| mix50_seed4011 | 0.5 | 1.7799 | 15/25 | 9/25 | NO/reference |
| mix00_seed4027 | 0.0 | 1.7773 | 9/25 | 7/25 | NO/reference |
| mix25_seed4027 | 0.25 | 1.7781 | 17/25 | 9/25 | NO/reference |
| mix50_seed4027 | 0.5 | 1.7784 | 13/25 | 8/25 | NO/reference |

**Pre-frozen model choice:** `base`. Feasible constraint-trained adapters (before final test): []. Promotion constraints were: strict official dev compliance no more than one question below original base, at most two extra 384-token truncations, and Dolly validation CE at least 0.10 nats/token lower than base. If none feasible choose original base; this is a normal negative outcome, not a reason to invent a winner.

## Independent final Google IFEval60: no model choice after observing final test

| Frozen model | Official strict pass /60 | Strict all-instruction prompt accuracy | Loose prompt accuracy | Reached 384 output cap |
|---|---:|---:|---:|---:|
| base | 47/60 | 78.3% | 80.0% | 25.0% |
| old_full | 38/60 | 63.3% | 63.3% | 36.7% |

### Exploratory bootstrap CI for selected minus comparator strict success rate
| Comparator | Selected−comparator within-family fixed-test 95% CI |
|---|---:|
| base | [+0.0, +0.0] percentage points |
| old_full | [+3.3, +26.7] percentage points |

**Interpretation and limits:** Real format demonstrations do not guarantee improved rule-following. The independent test contains only **60 public IFEval prompts from five categories**, selected from remaining keys after excluding all Stage36/38 observed prompts (240 keys). Not full 541 IFEval, not a natural production user population. Two training seeds per replay share cannot establish robust generality. Training-data transformation into JSON/XML/markers is deterministic but not independently human reviewed. Original model could have seen public Dolly/IFEval during pretraining; exact source and test hashes remain in the manifest.

Only validation-selected best trained adapter is published, if any passes the pre-frozen rule; all other five original adapter files remain on Tang with their verified Git blob hashes and fully reproducing train source/config/logs. No unsupported claim that a test-driven model is safe, semantically correct, human-aligned or suitable for rollout.
