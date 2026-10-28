# Stage 8 multilingual Kubernetes documentation: provenance and license

This dataset is composed of **adapted excerpts of Kubernetes website documentation**, not newly authored Kubernetes specifications or production traffic. The source is the official [kubernetes/website](https://github.com/kubernetes/website) repository, exact upstream Git commit `ed70e311737a341574ed5758b7af24e3bba232cb` (2026-10-26).

**Original authors:** The Kubernetes Authors and contributing localization teams.

**Source license:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Use of the Kubernetes name does not imply affiliation or endorsement.

**Changes:** We stripped YAML frontmatter, code fences, inline website boilerplate, Markdown headings/links and site shortcodes; normalized whitespace and extracted two disjoint prose windows from each source document. These changes are documented; the benchmark should not be mistaken for the official translated content.

The four technical subject areas are Kubernetes Pods (workloads), Deployments (orchestration), Secrets (security/configuration) and Persistent Volumes (storage). Each area contains eight excerpts across English (`en`), Simplified Chinese (`zh-cn`) and Japanese (`ja`). This is **three languages within one cloud-native documentation domain**, not diverse financial, medical or social production traffic.

Every of the 24 JSONL records in `stage8_multilingual_k8s_docs.jsonl` contains the exact upstream pinned source URL, its Git blob SHA-1, the excerpt's SHA-256, topic and language. `stage8_multilingual_k8s_manifest.json` contains per-file source hashes, corpus SHA-256, license, attribution and transformations.

The original Markdown files are kept only in local ignored `results/stage8_source_documents`. To refresh or verify them, retrieve each source path at the pinned repository commit from `stage8_multilingual_k8s_manifest.json`, then run:

```bash
python3 scripts/build_stage8_corpus.py --verify-existing
```

A source change fails its pinned Git blob SHA check, rather than silently updating the benchmark. Distribution or adaptation requires retaining attribution and the CC BY 4.0 license URL; different jurisdictions may impose additional requirements.

For Stage8 serving experiments, **actual request prompt_tokens from vLLM** are required; dataset character counts are not evidence of token length. The dataset is used for **inference-only performance studies**, not model training.
