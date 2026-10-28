"""Stage8 prompt selection and tokenizer calibration offline contract."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.freeze_stage8_requests import (
    BUCKET_TOKENS, INSTRUCTIONS, assign_docs, build_prompt, freeze_doc,
)

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/"data/stage8_multilingual_k8s_docs.jsonl"


class _ApproxTokenizer:
    """Fake tokenizer is ONLY for deterministic unit-test logic, not benchmarking."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text=messages[0]["content"]
        return [0]*max(1,(len(text)//4))


class Stage8FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=[json.loads(x) for x in CORPUS.read_text().splitlines()]

    def test_balanced_language_and_tokens(self):
        jobs=assign_docs(self.data)
        self.assertEqual(len(jobs),24)
        self.assertEqual(len({x["doc_id"] for x in jobs}),24)
        for lang in ("en","zh-cn","ja"):
            lang_rows=[x for x in jobs if x["language"]==lang]
            self.assertEqual(len(lang_rows),8)
            self.assertEqual({b:sum(x["bucket"]==b for x in lang_rows)
                              for b in BUCKET_TOKENS},{x:2 for x in BUCKET_TOKENS})
        self.assertEqual([x["doc_id"] for x in jobs],
                         [x["doc_id"] for x in assign_docs(self.data)])

    def test_fake_tokenizer_checks_extraction_logic_only(self):
        job=next(x for x in assign_docs(self.data) if x["bucket"]=="medium")
        result=freeze_doc(job,_ApproxTokenizer())
        self.assertEqual(result["target_prompt_tokens"],300)
        self.assertLessEqual(abs(result["calibrated_prompt_tokens"]["label"]-300),3)
        self.assertEqual(hashlib.sha256(result["prompt_text"].encode()).hexdigest(),
                         result["prompt_sha256"])

    def test_different_instructions_use_same_frozen_excerpt(self):
        job=next(x for x in assign_docs(self.data) if x["bucket"]=="short")
        x=freeze_doc(job,_ApproxTokenizer())
        self.assertEqual(set(x["calibrated_prompt_tokens"]),set(INSTRUCTIONS))
        self.assertGreater(x["prefix_characters"],0)


if __name__=="__main__":
    unittest.main()
