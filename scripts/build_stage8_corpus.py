"""Create a pinned multilingual technical documentation benchmark.

Source: Kubernetes website (CC BY 4.0); 4 cloud-native technical topics,
English, Chinese and Japanese. Two disjoint natural text windows from each
source. This is NOT a representative cross-industry/production traffic sample.
Original upstream source files live only under ignored results/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SOURCE_REPO = "https://github.com/kubernetes/website"
SOURCE_COMMIT = "ed70e311737a341574ed5758b7af24e3bba232cb"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
TOPICS = {
    "pods": "concepts/workloads/pods/_index.md",
    "deployment": "concepts/workloads/controllers/deployment.md",
    "secrets": "concepts/configuration/secret.md",
    "persistent-volumes": "concepts/storage/persistent-volumes.md",
}
LANGUAGES = ("en", "zh-cn", "ja")
# Git blob IDs from the pinned upstream commit, not loose unverified URLs.
BLOBS = {
    ("pods", "en"): "7ef439c12e8707771d29be3a454e9115ecde143c",
    ("pods", "zh-cn"): "ded431746812ce3b05fd2a71a6ae9982326d2f46",
    ("pods", "ja"): "b72a52fcd1a2040ea6a81b7f30d208c0635ee53c",
    ("deployment", "en"): "4e4d7dde8cdc6b41bcd26875286b0bc673b14d2b",
    ("deployment", "zh-cn"): "f800d9feb8071c4b04d96ab050fa7c38d86e6714",
    ("deployment", "ja"): "43e0046934b234dc9be25bc379622ee3829a6de9",
    ("secrets", "en"): "f10717fc581d4d149a958dc32c6e4b5ff9531218",
    ("secrets", "zh-cn"): "c29b48dd68ed9b8302788f262f9686f84400a9ca",
    ("secrets", "ja"): "c08b86ec835ac4a4e7538bd4968cc34bd23c292a",
    ("persistent-volumes", "en"): "ea0880a3f91d33553f6630a5f350330429a41cd1",
    ("persistent-volumes", "zh-cn"): "18723d0d1ca593a2b19750d4b6e2a3c06e20a86d",
    ("persistent-volumes", "ja"): "427706e544cfa45fdc8bba2ef28811dab6e94472",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\x00" + data).hexdigest()


def normalize_markdown(source: str) -> str:
    """Remove web boilerplate and code fences; keep editorial natural prose."""
    source = re.sub(r"(?s)\A---\s*\n.*?\n---\s*\n", "", source)
    source = re.sub(r"(?s)`{3,}.*?`{3,}", " ", source)
    source = re.sub(r"(?m)^\s*{{[<%].*?[>%]}}\s*$", "", source)
    source = re.sub(r"(?m)^\s*(?:\|.+|#{1,6}\s+.*)$", " ", source)
    source = re.sub(r"{{[<%].*?[>%]}}", " ", source)
    source = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", source)
    source = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", source)
    source = re.sub(r"<[^>]+>", " ", source)
    source = re.sub(r"https?://[^\s)]+", " ", source)
    source = re.sub(r"\s+", " ", source).strip()
    if len(source) < 6000:
        raise ValueError("not enough text for two disjoint windows")
    return source


def windows(text: str) -> list[dict]:
    """Two nonoverlapping 4k-character windows from separate document regions."""
    length = min(4000, int(len(text) * .35))
    assert length >= 2100
    spans = []
    for fraction in (.07, .57):
        start = int(len(text) * fraction)
        end = start + length
        if end > len(text):
            raise ValueError("excerpt exceeds body length")
        # Keep windows from the original prose in fixed positions; no edits.
        s = text[start:end].strip()
        spans.append({
            "start_character": start,
            "end_character": end,
            "text": s,
            "text_sha256": digest(s.encode("utf-8")),
            "text_characters": len(s),
        })
    if spans[0]["end_character"] >= spans[1]["start_character"]:
        raise ValueError("source excerpts overlap")
    return spans


def build(src_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    sources = []
    for topic, relative in TOPICS.items():
        for lang in LANGUAGES:
            filename = f"{topic}_{lang}.md"
            raw = (src_dir / filename).read_bytes()
            expected = BLOBS[(topic, lang)]
            actual = git_blob_sha(raw)
            if actual != expected:
                raise ValueError(f"upstream Git blob identity mismatched {filename}: {actual} != {expected}")
            body = normalize_markdown(raw.decode("utf-8"))
            sources.append({
                "topic": topic,
                "language": lang,
                "upstream_path": f"content/{lang}/docs/{relative}",
                "upstream_url": f"{SOURCE_REPO}/blob/{SOURCE_COMMIT}/content/{lang}/docs/{relative}",
                "git_blob_sha1": actual,
                "raw_sha256": digest(raw),
                "raw_bytes": len(raw),
                "normalized_body_characters": len(body),
                "license": "CC BY 4.0",
                "attribution": "The Kubernetes Authors and contributing translators",
                "modifications": "YAML frontmatter / code fences / headings removed and two fixed plain-text windows excerpted",
            })
            for part, snippet in enumerate(windows(body)):
                rows.append({
                    "doc_id": f"{topic}-{lang}-part{part}",
                    "topic": topic,
                    "language": lang,
                    "source_commit": SOURCE_COMMIT,
                    "source_git_blob": actual,
                    "source_url": sources[-1]["upstream_url"],
                    "license": "CC BY 4.0",
                    "part": part,
                    **snippet,
                })
    if len(rows) != 24 or len({x["doc_id"] for x in rows}) != 24:
        raise AssertionError("expected 24 distinct excerpt identities")
    assert {x["language"] for x in rows} == set(LANGUAGES)
    assert {x["topic"] for x in rows} == set(TOPICS)
    manifest = {
        "schema": "stage8-kubernetes-multilingual-docs-v1",
        "source_repo": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "license": "CC BY 4.0",
        "license_url": LICENSE,
        "attribution": "The Kubernetes Authors and contributing translators",
        "modifications": "Removed site metadata/fences/headings and excerpted disjoint natural-text spans; no upstream changes",
        "languages": LANGUAGES,
        "topics": tuple(TOPICS),
        "source_document_count": len(sources),
        "excerpt_count": len(rows),
        "source_raw_not_committed": True,
        "not_production_or_cross_industry": True,
        "sources": sources,
        "source_sha256_by_excerpt": {x["doc_id"]: x["text_sha256"] for x in rows},
        "citation_note": "Each excerpt's source_url and license must be retained with copies of this data.",
    }
    return rows, manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=Path("results/stage8_source_documents"))
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--verify-existing", action="store_true")
    args = p.parse_args(argv)
    rows, manifest = build(args.source_dir)
    data = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    manifest["corpus_jsonl_sha256"] = digest(data.encode("utf-8"))
    metadata = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    output = args.output_dir / "stage8_multilingual_k8s_docs.jsonl"
    index = args.output_dir / "stage8_multilingual_k8s_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.verify_existing:
        if output.read_text(encoding="utf-8") != data or index.read_text(encoding="utf-8") != metadata:
            raise ValueError("existing frozen Stage8 corpus differs from pinned source")
        print("VERIFIED_STAGED8_CORPUS", len(rows), manifest["corpus_jsonl_sha256"])
        return 0
    if output.exists() or index.exists():
        raise FileExistsError("frozen corpus exists; use --verify-existing")
    output.write_text(data, encoding="utf-8")
    index.write_text(metadata, encoding="utf-8")
    print("FROZEN_STAGE8", len(rows), len(data.encode("utf-8")), manifest["corpus_jsonl_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
