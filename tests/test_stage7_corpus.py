"""Stage7 corpus is frozen, attributed and structurally nonoverlapping."""
from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"


class Stage7CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[json.loads(line) for line in (DATA/"stage7_public_domain_passages.jsonl").read_text().splitlines()]
        cls.manifest=json.loads((DATA/"stage7_public_domain_manifest.json").read_text())

    def test_30_distinct_passages_from_five_books(self):
        self.assertEqual(len(self.rows),30)
        self.assertEqual(len({x["doc_id"] for x in self.rows}),30)
        self.assertEqual(len({x["text_sha256"] for x in self.rows}),30)
        self.assertEqual(len({x["source_ebook_id"] for x in self.rows}),5)
        self.assertEqual(set(Counter(x["source_ebook_id"] for x in self.rows).values()),{6})
        self.assertTrue(all(9200<=x["text_chars"]<=10500 for x in self.rows))

    def test_manifest_pins_sha256_and_source_provenance(self):
        m=self.manifest
        self.assertEqual(m["schema"],"stage7-public-domain-nonrepeating-corpus-v1")
        self.assertEqual(m["document_count"],30)
        self.assertEqual(m["book_count"],5)
        corpus=(DATA/"stage7_public_domain_passages.jsonl").read_bytes()
        self.assertEqual(hashlib.sha256(corpus).hexdigest(),m["corpus_jsonl_sha256"])
        expected={r["doc_id"]:r["text_sha256"] for r in m["source_excerpts"]}
        for row in self.rows:
            self.assertEqual(hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                             row["text_sha256"])
            self.assertEqual(expected[row["doc_id"]],row["text_sha256"])
            self.assertNotIn("PROJECT GUTENBERG LICENSE",row["text"].upper())
            self.assertNotIn("*** START",row["text"])

    def test_all_attributed_and_us_copyright_scope(self):
        self.assertTrue("USA" in self.manifest["rights_note"] or "US" in self.manifest["rights_note"])
        for source in self.manifest["sources"]:
            self.assertTrue(source["catalog_url"].startswith("https://www.gutenberg.org/ebooks/"))
            self.assertEqual(len(source["source_sha256"]),64)
            self.assertIn("public domain",source["rights_catalog"])
        self.assertTrue(self.manifest["source_files_not_committed"])


if __name__=="__main__":
    unittest.main()
