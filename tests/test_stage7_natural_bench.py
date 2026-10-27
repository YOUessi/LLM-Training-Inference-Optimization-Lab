"""Stage7 genuine public-domain prose workload, no CUDA required."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage7_natural_bench import (
    BUCKETS, GENERATION, by_bucket, load_corpus, parse_args,
    truncate_sentence, workload,
)

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/"data/stage7_public_domain_passages.jsonl"
MANIFEST=ROOT/"data/stage7_public_domain_manifest.json"


class Stage7NaturalWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.meta=load_corpus(CORPUS,MANIFEST)

    def test_five_genres_four_lengths_24_unique_docs(self):
        items,_=workload(self.rows,seed=42)
        self.assertEqual(len(items),24)
        self.assertEqual(len({x["doc_id"] for x in items}),24)
        self.assertEqual(len({x["genre"] for x in items}),5)
        for bucket in BUCKETS:
            self.assertEqual(sum(x["bucket"]==bucket for x in items),6)
        self.assertEqual(len(set(x["prompt_sha256"] for x in items)),24)
        self.assertTrue(all("Project Gutenberg License" not in x["prompt"] for x in items))

    def test_two_seed_orders_are_reproducible_but_different(self):
        a,_=workload(self.rows,seed=42)
        b,_=workload(self.rows,seed=42)
        c,_=workload(self.rows,seed=137)
        self.assertEqual([x["doc_id"] for x in a],[x["doc_id"] for x in b])
        self.assertNotEqual([x["doc_id"] for x in a],[x["doc_id"] for x in c])

    def test_actual_generation_token_gates_preregistered(self):
        self.assertEqual(GENERATION["short"]["max_new_tokens"],32)
        self.assertEqual(GENERATION["long"]["max_new_tokens"],128)
        self.assertGreaterEqual(GENERATION["long"]["min_mean_generated_tokens"],85)
        # Offline Qwen calibration: corrected long directive fits the original gates.
        self.assertLess(len(GENERATION["long"]["response_instruction"]),140)
        self.assertIn("25 detailed numbered",GENERATION["long"]["response_instruction"])

    def test_bucket_fails_closed_for_invalid_provider_tokens(self):
        cases=[]
        for name,(_,low,up) in BUCKETS.items():
            for i in range(6):
                cases.append({
                    "bucket":name,"status":"ok","prompt_tokens":(low+up)//2,
                    "completion_tokens":21,"ttft_ms":10.+i,
                    "latency_ms":100.+i,
                })
        out=by_bucket(cases)
        self.assertTrue(all(x["status"]=="pass" for x in out.values()))
        cases[-1]["prompt_tokens"]=None
        self.assertEqual(by_bucket(cases)["very_long"]["status"],"invalid")
        cases[-1]["prompt_tokens"]=9999
        self.assertEqual(by_bucket(cases)["very_long"]["status"],"invalid")

    def test_tampered_frozen_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.jsonl"
            p.write_text(CORPUS.read_text()+" ",encoding="utf-8")
            with self.assertRaises(ValueError):
                load_corpus(p,MANIFEST)

    def test_parser_validates_seed_and_concurrency(self):
        good=parse_args(["--model","qwen","--mode","long","--seed","137",
                         "--concurrency","8","--output","/tmp/a.json"])
        self.assertEqual(good.mode,"long")
        for extra in (["--seed","21"],["--concurrency","16"],
                      ["--mode","invalid"],["--warmup","4"]):
            with self.subTest(extra=extra),self.assertRaises(SystemExit):
                parse_args(["--model","qwen","--mode","short",
                            "--seed","42","--concurrency","1",
                            "--output","/tmp/x.json"]+extra)


if __name__=="__main__":
    unittest.main()
