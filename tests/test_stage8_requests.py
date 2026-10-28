"""Frozen tokenizer-calibrated request attribution and hash consistency."""
from __future__ import annotations

import collections
import hashlib
import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"data"
REQUESTS=ROOT/"stage8_frozen_requests.jsonl"
MANIFEST=ROOT/"stage8_frozen_requests_manifest.json"


class Stage8FrozenRequestsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[json.loads(x) for x in REQUESTS.read_text(encoding="utf-8").splitlines()]
        cls.manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_tokenized_version_and_source_match(self):
        m=self.manifest
        old=json.loads((ROOT/"stage8_multilingual_k8s_manifest.json").read_text())
        self.assertEqual(m["schema"],"stage8-frozen-qwen-token-calibrated-v1")
        self.assertEqual(m["original_corpus_sha256"],old["corpus_jsonl_sha256"])
        self.assertEqual(m["tokenizer_snapshot"],
                         "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
        self.assertEqual(hashlib.sha256(REQUESTS.read_bytes()).hexdigest(),
                         m["request_jsonl_sha256"])

    def test_language_and_length_balance(self):
        self.assertEqual(len(self.rows),24)
        self.assertEqual(len({r["request_id"] for r in self.rows}),24)
        self.assertEqual(dict(collections.Counter(r["language"] for r in self.rows)),
                         {"en":8,"zh-cn":8,"ja":8})
        self.assertEqual(dict(collections.Counter(r["bucket"] for r in self.rows)),
                         {"short":6,"medium":6,"long":6,"very_long":6})
        for r in self.rows:
            t=r["target_prompt_tokens"]
            for count in r["calibrated_prompt_tokens"].values():
                self.assertLessEqual(abs(count-t),95)
                self.assertLess(count,2000)
            self.assertEqual(hashlib.sha256(r["prompt_text"].encode("utf-8")).hexdigest(),
                             r["prompt_sha256"])

    def test_all_instructions_frozen(self):
        self.assertEqual(set(self.manifest["generation_instructions"]),
                         {"label","summary","detailed"})
        self.assertTrue(self.manifest["not_a_training_dataset"])
        self.assertEqual(self.manifest["request_count"],24)


if __name__=="__main__":
    unittest.main()
