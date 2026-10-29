# Stage29: frozen test paired base vs selected Qwen3 SFT generation quality

Preserve Stage26 selection: **only** selected B (Rank16 BF16 96-step 8B QLoRA) vs original unmodified Qwen3-8B baseline. All 96 Dolly frozen test examples from source SHA and original CC BY-SA attribution; no new training, tuning, or candidate selection based on test results. Deterministic Qwen3 chat template (thinking disabled), greedy max128 new tokens. Evaluate each prompt with base adapters disabled and selected adapter enabled, preserving exact input identities. Report normalized exact match, token-overlap F1, word ROUGE-L F1, reference/generation lengths, paired per-category metrics and hash per output, never substitute only-accepted averages. Save no full copyrighted reference answer duplicates beyond original properly attributed dataset manifest. All GPU work run under source-pinned shared mutex, output no-overwrite and fail closed.

Token overlap/ROUGE-L compare to a single reference and **do not establish semantic/factual correctness, instruction following, safety or human preference**. We should not claim comprehensive LLM alignment from this alone. This is supplementary quality context to the primary 96-heldout assistant CE result.

### Rerun amendment
The first frozen script `9f35956` exited before any GPU work with a Bash parser bug (`|| {tail` lacking whitespace). Its exact log and source are archived in `evidence/stage29_invalid_v1`. The corrected rerun uses a new pinned SHA; no old results exist or are merged.

### Attention-mask repeatability amendment
The v2 source `3e83f20` was stopped during first generation on the HuggingFace warning that `pad_token_id==eos_token_id` does not allow inference of an attention mask. Single-example prompts were unpadded but the complete 96-case measurement is rerun with an **explicit all-ones attention mask**, removing an ambiguity that might affect evaluation. Original interrupted log and source retained in `evidence/stage29_invalid_v2`. The previous run made no output JSON and is not used.
