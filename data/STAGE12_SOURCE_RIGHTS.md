# Stage12 independent-domain natural text workload

Stage12 does NOT use Kubernetes documentation. This held-out input corpus
contains 24 source- and tokenizer-pinned excerpts from the Stage7 natural
prose corpus of five publicly attributed classic works: social fiction,
science fiction, fantasy, travel fiction and natural science.

The exact source eBook URLs, original author names, Gutenberg full-file
SHA256 fingerprints, excerpt SHA256 and rights notes are recorded in
data/stage7_public_domain_manifest.json and data/README.md.

The original works were catalogued by Project Gutenberg as public domain
in the **United States**. Other jurisdictions may apply different rules;
Project Gutenberg does not sponsor or endorse this research.

Stage12 freezes the exact Qwen2.5-1.5B-Instruct tokenizer snapshot, six
prompts at each actual input length target 170 / 300 / 470 / 680 Tokens
and 24 unique request identifiers. Every request includes its source
excerpt SHA256 and a distinct prompt SHA256.

These are **English-language natural literary/science passages**, not
modern enterprise traffic and not independent multilingual evidence.
The held-out benefit is the new *source-content domain* relative to
Kubernetes-serving experiments; statistical model generalization is not
claimed until independent real GPU tests pass.

No model weights are trained or fine-tuned from this corpus. Source
text and citations are stored for reproducibility of an inference-only
performance benchmark.

Manifest: data/stage12_literary_requests_manifest.json
Frozen prompts: data/stage12_literary_requests.jsonl
