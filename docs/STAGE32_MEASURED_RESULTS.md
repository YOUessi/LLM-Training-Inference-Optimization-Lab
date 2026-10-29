# Stage32: independent mDeBERTa-v3 XNLI reference-agreement proxy on 64 frozen Dolly test inputs

**Actual measured**, 64 unique frozen test prompts (exactly 8 from each of 8 Dolly categories), 3 Qwen3-8B NF4 variants — original base, prior Stage26 short QLoRA, and Stage31 validation-selected full512 QLoRA. 64×3 deterministic greedy generations and **384 independent directional NLI cross-encoder scores**, generated with fixed non-thinking Qwen chat templates, max 128 output tokens. The semantic judge is locally cached **MoritzLaurer/mDeBERTa-v3-base-mnli-xnli**, snapshot `8adb042d...` — independent of the Qwen weights; original licensed human reference answers and input hashes are in frozen Stage26. Raw per-example generated outputs, generation SHA, class-wise scores, bidirectional entailment, contradiction and 2,000 stratified bootstrap resamples in `evidence/stage32_semantic/`.

| Reference-oriented proxy | Original 8B | Stage26 short QLoRA | Stage31 full QLoRA |
|---|---:|---:|---:|
| Mean two-direction NLI entailment | 0.2085 | 0.3274 | **0.3435** |
| Word overlap F1 | 0.2926 | 0.4178 | **0.4429** |
| Reference-contradicts-response probability | 0.0587 | 0.1739 | 0.1633 |

Within-fixed-set stratified bootstrap CI for the **Stage31 minus Stage26 short-SFT bidirectional entailment difference**: [-0.0360, +0.0710], crossing zero. **No established semantic improvement of full SFT relative to short SFT** on this proxy. Scores against the base improve but the reference-to-answer contradiction proxy also rises relative to base; do not hide this potentially adverse trend. NLI is trained on a different inference task, and public Dolly open-ended answers may have legitimate paraphrases, creative choices and differing lengths; it is not a human correctness, hallucination or safety verdict.

| Dolly source category | Base NLI mean | Short QLoRA NLI mean | Full QLoRA NLI mean |
|---|---:|---:|---:|
| brainstorming | 0.041 | 0.141 | 0.248 |
| classification | 0.261 | 0.739 | 0.587 |
| closed_qa | 0.404 | 0.449 | 0.463 |
| creative_writing | 0.043 | 0.199 | 0.225 |
| general_qa | 0.074 | 0.115 | 0.142 |
| information_extraction | 0.512 | 0.369 | 0.451 |
| open_qa | 0.186 | 0.333 | 0.313 |
| summarization | 0.147 | 0.275 | 0.319 |

No selection used this test; the same selected Stage31 model was retained. There is **no external natural-user traffic, human review, or independent-domain quality benchmark** in this result.

