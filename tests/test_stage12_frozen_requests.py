"""Validate real Qwen-tokenizer-frozen non-Kubernetes Stage12 requests."""
from __future__ import annotations

import collections
import hashlib
import json
import unittest
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]/"data"
FILE=ROOT/"stage12_literary_requests.jsonl"
META=ROOT/"stage12_literary_requests_manifest.json"


class Stage12FrozenRequestsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[json.loads(x) for x in FILE.read_text(encoding="utf-8").splitlines()]
        cls.meta=json.loads(META.read_text(encoding="utf-8"))

    def test_source_and_tokenizer_integrity(self):
        m=self.meta
        self.assertEqual(m["schema"],"stage12-independent-domain-natural-prose-v1")
        self.assertEqual(m["request_count"],24)
        self.assertEqual(m["original_corpus_sha256"],
                         "2912a4d8cc17789a1b6b339cac927a27ec41f24f7de140d84466ecd441e095ca")
        self.assertEqual(m["tokenizer_snapshot"],
                         "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
        self.assertEqual(hashlib.sha256(FILE.read_bytes()).hexdigest(),m["request_jsonl_sha256"])
        self.assertTrue(m["not_training_data"])

    def test_unique_docs_and_five_genres(self):
        a=self.rows
        self.assertEqual(len(a),24)
        self.assertEqual(len({x["request_id"] for x in a}),24)
        self.assertEqual(len({x["topic"] for x in a}),5)
        self.assertEqual(set(collections.Counter(x["topic"] for x in a).values()),{4,5})
        self.assertEqual(dict(collections.Counter(x["bucket"] for x in a)),
                         {"short":6,"medium":6,"long":6,"very_long":6})
        self.assertEqual({x["language"] for x in a},{"en"})

    def test_all_prompt_hashes_and_real_model_calibration(self):
        for x in self.rows:
            self.assertEqual(hashlib.sha256(x["prompt_text"].encode()).hexdigest(),
                             x["prompt_sha256"])
            self.assertEqual(len(x["source_excerpt_sha256"]),64)
            self.assertNotIn("Kubernetes",x["prompt_text"])
            self.assertTrue(all(abs(v-x["target_prompt_tokens"])<=95
                                for v in x["calibrated_prompt_tokens"].values()))
            self.assertLess(x["calibrated_prompt_tokens"]["label"],750)

    def test_manifest_matches_serving_modes(self):
        self.assertEqual(set(self.meta["generation_instructions"]),
                         {"label","summary","detailed"})
        self.assertEqual(self.meta["language_counts"],{"en":24})


if __name__=="__main__":
    unittest.main()
