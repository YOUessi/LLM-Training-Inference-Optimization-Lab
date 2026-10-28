"""Stage12 non-Kubernetes admission workload selection, CPU-only invariants."""
from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.freeze_stage12_literary_requests import (
    assign,INSTRUCTIONS,LENGTHS,TARGETS,body_prompt
)


CORPUS=Path(__file__).resolve().parents[1]/"data/stage7_public_domain_passages.jsonl"


class Stage12IndependentDomainPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[json.loads(x) for x in CORPUS.read_text().splitlines()]

    def test_24_unique_and_5_genres_balanced(self):
        r=assign(self.rows)
        self.assertEqual(len(r),24)
        self.assertEqual(len({x["doc_id"] for x in r}),24)
        self.assertEqual(len({x["genre"] for x in r}),5)
        self.assertEqual(set(Counter(x["genre"] for x in r).values()),{4,5})
        self.assertEqual(dict(Counter(x["bucket"] for x in r)),{x:6 for x in LENGTHS})
        self.assertEqual([x["doc_id"] for x in r],
                         [x["doc_id"] for x in assign(self.rows)])

    def test_generation_instructions_match_serving_interface(self):
        self.assertEqual(set(INSTRUCTIONS),{"label","summary","detailed"})
        self.assertEqual(TARGETS,{"short":170,"medium":300,"long":470,"very_long":680})
        self.assertNotIn("Kubernetes",INSTRUCTIONS["label"])
        self.assertNotIn("Kubernetes",INSTRUCTIONS["detailed"])

    def test_prompt_is_derived_from_pinned_excerpt_without_mutation(self):
        x=assign(self.rows)[0]
        snippet=x["text"][:175]
        text=body_prompt(x,snippet)
        self.assertIn(snippet,text)
        self.assertIn(x["doc_id"],text)
        self.assertIn(x["genre"],text)
        self.assertNotIn("PROJECT GUTENBERG LICENSE",text.upper())

    def test_incorrect_row_count_rejected(self):
        with self.assertRaises(ValueError):
            assign(self.rows[:20])


if __name__=="__main__":
    unittest.main()
