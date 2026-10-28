"""Verify frozen CC BY 4.0 multilingual corpus source metadata and integrity."""
from __future__ import annotations

import collections
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data"
FILE = ROOT / "stage8_multilingual_k8s_docs.jsonl"
MANIFEST = ROOT / "stage8_multilingual_k8s_manifest.json"


class Stage8CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(x) for x in FILE.read_text(encoding="utf-8").splitlines()]
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_24_unique_windows_across_language_and_topic(self):
        r = self.rows
        self.assertEqual(len(r), 24)
        self.assertEqual(len({x["doc_id"] for x in r}), 24)
        self.assertEqual(len({x["text_sha256"] for x in r}), 24)
        self.assertEqual(dict(collections.Counter(x["language"] for x in r)),
                         {"en": 8, "zh-cn": 8, "ja": 8})
        self.assertEqual(set(collections.Counter(x["topic"] for x in r).values()), {6})
        self.assertTrue(all(2600 <= x["text_characters"] <= 4000 for x in r))

    def test_all_passage_and_full_corpus_hashes(self):
        m = self.manifest
        self.assertEqual(m["schema"], "stage8-kubernetes-multilingual-docs-v1")
        self.assertEqual(hashlib.sha256(FILE.read_bytes()).hexdigest(),
                         m["corpus_jsonl_sha256"])
        self.assertEqual(len(m["sources"]), 12)
        for r in self.rows:
            self.assertEqual(hashlib.sha256(r["text"].encode("utf-8")).hexdigest(),
                             r["text_sha256"])
            self.assertEqual(m["source_sha256_by_excerpt"][r["doc_id"]],
                             r["text_sha256"])

    def test_attribution_and_exact_upstream_commit(self):
        m = self.manifest
        self.assertEqual(m["source_commit"],
                         "ed70e311737a341574ed5758b7af24e3bba232cb")
        self.assertEqual(m["license"], "CC BY 4.0")
        self.assertIn("creativecommons.org/licenses/by/4.0", m["license_url"])
        self.assertTrue(m["source_raw_not_committed"])
        self.assertTrue(m["not_production_or_cross_industry"])
        for row in m["sources"]:
            self.assertIn(m["source_commit"], row["upstream_url"])
            self.assertEqual(len(row["raw_sha256"]), 64)
            self.assertEqual(len(row["git_blob_sha1"]), 40)
            self.assertIn("Kubernetes Authors", row["attribution"])

    def test_source_windows_do_not_overlap(self):
        by_source = collections.defaultdict(list)
        for r in self.rows:
            by_source[(r["topic"], r["language"])].append(r)
        for source, rows in by_source.items():
            with self.subTest(source=source):
                self.assertEqual(len(rows), 2)
                rows.sort(key=lambda x: x["start_character"])
                self.assertLess(rows[0]["end_character"],
                                rows[1]["start_character"])


if __name__ == "__main__":
    unittest.main()
