# Stage 7 — Frozen public-domain prose corpus

This is a reproducible **multi-genre language workload**, not a proprietary or
representative enterprise traffic dataset. It is used to test vLLM prefill,
mixed decoding lengths, queuing, TTFT and throughput on real GPU hardware.

## Provenance and rights

The benchmark contains 30 non-overlapping, approximately 10,000-character
passages sampled from five English-language public-domain classics. The
selected Project Gutenberg catalog entries are marked **public domain in the
United States**, not a worldwide blanket permission. Consult the law of any
other location before redistributing. Project Gutenberg is a registered
trademark; this research is not sponsored or endorsed by Project Gutenberg.

| eBook | Original work | Author | Workload genre |
|---:|---|---|---|
| [1342](https://www.gutenberg.org/ebooks/1342) | Pride and Prejudice | Jane Austen | Social fiction |
| [84](https://www.gutenberg.org/ebooks/84) | Frankenstein | Mary Shelley | Science fiction |
| [2009](https://www.gutenberg.org/ebooks/2009) | The Origin of Species | Charles Darwin | Natural science |
| [103](https://www.gutenberg.org/ebooks/103) | Around the World in Eighty Days | Jules Verne | Travel fiction |
| [11](https://www.gutenberg.org/ebooks/11) | Alice's Adventures in Wonderland | Lewis Carroll | Fantasy |

Source/rights reference:
https://www.gutenberg.org/policy/permission

## Building/verifying

The full text files (source books) are *not* committed to GitHub.
They live locally in the ignored `results/stage7_gutenberg_downloads`.
Download `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`
for each of the five IDs before building.

```bash
python3 scripts/build_stage7_corpus.py
python3 scripts/build_stage7_corpus.py --verify-existing
```

The builder strictly checks the **exact SHA-256 of each full original file**
and refuses silent regeneration after an upstream edition changes. It removes
Gutenberg legal header/footer material and uses deterministic source offsets
at approximately 10%, 24%, 38%, 52%, 66% and 80% of each body's prose,
keeping sentence boundaries.

Published files:

- `data/stage7_public_domain_passages.jsonl`: the 30 frozen excerpts and
  individual SHA-256 digests
- `data/stage7_public_domain_manifest.json`: source URLs, rights statement,
  source-file SHA-256, passage-level SHA-256 and corpus-file SHA-256

**Validation:** source text is nonrepeating within and across the experiment's
frozen passages by construction (disjoint windows). Do not interpret this as
an adversarial plagiarism detector or guarantee of zero repeated vocabulary.
Actual model input length is measured via provider-reported prompt_tokens,
not merely character counts.

Project Gutenberg is not affiliated with this benchmark. No book text is
used for training or model fine-tuning; all use is inference-only measurement.
