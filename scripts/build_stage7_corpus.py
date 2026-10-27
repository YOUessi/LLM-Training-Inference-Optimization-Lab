"""Freeze multi-genre, nonrepeating Project Gutenberg prose excerpts.

This is a data-construction script, not an inference benchmark. The full
original source files are downloaded into ignored results/ (not published).
A pinned sha256 guards against silent Gutenberg edition updates.

Works are catalogued as US public domain; verify local law before republishing
outside the US. Project Gutenberg name/marks are not endorsements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SOURCES = (
    (1342, "Pride and Prejudice", "Jane Austen", "social fiction",
     "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75"),
    (84, "Frankenstein", "Mary Shelley", "science fiction",
     "2b97b7f8aad508649481d8dd4f6b54f9b965ddb3ba634e9151c815f537b49e42"),
    (2009, "The Origin of Species", "Charles Darwin", "natural science",
     "2b13c8524d813e706ad23336d81a142d48b44f1f96788b41494a0a83e1b82f88"),
    (103, "Around the World in Eighty Days", "Jules Verne", "travel fiction",
     "6f608ea2da50caf73245ad2996e4ee3c02ef1aa2152f7fcfe209df9aa39e7512"),
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll", "fantasy",
     "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e"),
)

OFFSETS = (.10, .24, .38, .52, .66, .80)
START = re.compile(r"(?im)^\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$")
END = re.compile(r"(?im)^\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*\s*$")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_gutenberg(raw: bytes) -> str:
    txt = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    first, last = START.search(txt), END.search(txt)
    if first is None or last is None or last.start() <= first.end():
        raise ValueError("Gutenberg license boilerplate delimiters missing")
    text = txt[first.end():last.start()]
    # Preserve natural sentence word order but remove edition-dependent wrapping.
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) < 100000:
        raise ValueError("book unexpectedly too short after cleanup")
    if "PROJECT GUTENBERG LICENSE" in normalized.upper():
        raise ValueError("must not freeze embedded Gutenberg license boilerplate")
    return normalized


def excerpt_at(text: str, fraction: float) -> str:
    pos = int(len(text) * fraction)
    # Move forward to the next clear sentence boundary.
    boundary = re.search(r"[.!?][”'\"]?\s+[A-Z]", text[pos:pos+400])
    if boundary is None:
        raise ValueError("no nearby natural sentence boundary")
    start = pos + boundary.end() - 1
    selected = text[start:start+11000]
    if len(selected) < 10000:
        raise ValueError("insufficient excerpt window")
    # Use the nearest previous full sentence boundary after about 9900 chars.
    good = list(re.finditer(r"[.!?][”'\"]?\s+[A-Z]", selected[9300:10400]))
    if not good:
        raise ValueError("no suitable sentence end boundary in excerpt")
    boundary_end = 9300 + good[-1].start() + 1
    clean = selected[:boundary_end].strip()
    if not 9200 <= len(clean) <= 10500:
        raise ValueError("excerpt size out of frozen contract")
    return clean


def build(source_dir: Path) -> tuple[list[dict], dict]:
    output = []
    sources = []
    for ebook_id, title, author, genre, expected_hash in SOURCES:
        source = source_dir / f"pg{ebook_id}.txt"
        raw = source.read_bytes()
        actual = sha(raw)
        if actual != expected_hash:
            raise ValueError(
                f"eBook {ebook_id} full-file digest changed: {actual}; "
                "do not silently regenerate with a new edition"
            )
        text = strip_gutenberg(raw)
        source_row = {
            "ebook_id": ebook_id, "title": title, "author": author, "genre": genre,
            "catalog_url": f"https://www.gutenberg.org/ebooks/{ebook_id}",
            "text_url": f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt",
            "source_sha256": expected_hash, "source_bytes": len(raw),
            "body_chars_after_boilerplate_removed": len(text),
            "rights_catalog": "Project Gutenberg: public domain in USA",
        }
        sources.append(source_row)
        for n, fraction in enumerate(OFFSETS):
            passage = excerpt_at(text, fraction)
            row = {
                "doc_id": f"pg{ebook_id}-section{n:02d}",
                "source_ebook_id": ebook_id, "genre": genre,
                "text": passage,
                "text_sha256": sha(passage.encode("utf-8")),
                "text_chars": len(passage),
                "excerpt_position_fraction": fraction,
            }
            output.append(row)
    ids = [x["doc_id"] for x in output]
    assert len(output) == len(SOURCES) * len(OFFSETS) == len(set(ids)) == 30
    # Near-duplicate guard across chunks and books.
    hashes = [sha(x["text"][:2000].encode("utf-8")) for x in output]
    assert len(hashes) == len(set(hashes)), "overlapping identical passage starts"
    manifest = {
        "schema": "stage7-public-domain-nonrepeating-corpus-v1",
        "capture_date": "2026-10-28",
        "book_count": len(SOURCES),
        "document_count": len(output),
        "passage_strategy": "six nonoverlapping approximate 9-10k-character full-sentence excerpts per source",
        "genres": sorted(set(x["genre"] for x in sources)),
        "source_files_not_committed": True,
        "rights_note": (
            "Source catalogs mark selected original works public domain in the USA. "
            "Check laws in other jurisdictions. The excerpts are attributed, and "
            "Project Gutenberg is not affiliated with this research."
        ),
        "corpus_description": (
            "Thirty frozen, varied passages from five published prose books; "
            "not a sample of production search traffic, and not guaranteed to "
            "match domain-specific private enterprise documents."
        ),
        "sources": sources,
        "source_excerpts": [
            {"doc_id": row["doc_id"],"text_sha256":row["text_sha256"],
             "text_chars":row["text_chars"],"source_ebook_id":row["source_ebook_id"]}
            for row in output
        ],
    }
    return output, manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=Path("results/stage7_gutenberg_downloads"))
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    p.add_argument("--verify-existing", action="store_true")
    a = p.parse_args(argv)
    data, meta = build(a.source_dir)
    out = a.output_dir / "stage7_public_domain_passages.jsonl"
    manifest = a.output_dir / "stage7_public_domain_manifest.json"
    a.output_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in data)
    meta["corpus_jsonl_sha256"] = sha(text.encode("utf-8"))
    metadata = json.dumps(meta,ensure_ascii=False,indent=2)+"\n"
    if a.verify_existing:
        if out.read_text(encoding="utf-8") != text or manifest.read_text(encoding="utf-8") != metadata:
            raise ValueError("frozen corpus changed from source input")
        print("FROZEN_CORPUS_VERIFIED",len(data),meta["corpus_jsonl_sha256"])
    else:
        if out.exists() or manifest.exists():
            raise FileExistsError("corpus exists; use --verify-existing")
        out.write_text(text,encoding="utf-8")
        manifest.write_text(metadata,encoding="utf-8")
        print("FROZEN_CORPUS_CREATED",len(data),len(text.encode("utf-8")),
              meta["corpus_jsonl_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
