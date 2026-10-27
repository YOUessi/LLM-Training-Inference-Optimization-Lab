"""CPU-only Stage7 real-data analyzer contract: source and workload coherence.

Fake test rows live only in TemporaryDirectory to prove fail-closed logic;
they are never published or described as GPU benchmarks.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage7_analysis import analyze
from llm_opt_lab.serving.stage7_natural_bench import (
    BUCKETS,GENERATION,load_corpus,workload
)

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/"data/stage7_public_domain_passages.jsonl"
MANIFEST=ROOT/"data/stage7_public_domain_manifest.json"


def fixture(root:Path) -> None:
    rows,manifest=load_corpus(CORPUS,MANIFEST)
    for seed in (42,137):
        items,_=workload(rows,seed=seed)
        for mode in ("short","long"):
            for concurrency in (1,2,4,8):
                records=[]
                group={}
                for bucket,(_,lo,hi) in BUCKETS.items():
                    group[bucket]={"status":"pass","sent":6,
                                   "min_actual_prompt_tokens":(lo+hi)//2,
                                   "max_actual_prompt_tokens":(lo+hi)//2,
                                   "p95_ttft_ms":50.}
                for index,doc in enumerate(items):
                    records.append({
                        "doc_id":doc["doc_id"],"bucket":doc["bucket"],
                        "genre":doc["genre"],"prompt_sha256":doc["prompt_sha256"],
                    })
                record={
                    "status":"measured",
                    "kind":"stage7_natural_public_domain_vllm_mixed_context",
                    "environment":{"git_commit":"a"*40},
                    "config":{
                        "seed":seed,"mode":mode,"concurrency":concurrency,
                        "max_new_tokens":GENERATION[mode]["max_new_tokens"],
                        "corpus_sha256":manifest["corpus_jsonl_sha256"],
                    },
                    "summary":{
                        "succeeded":24,"failed":0,"prompt_usage_coverage":1.,
                        "usage_coverage":1.,"p95_ttft_ms":50.+concurrency*25,
                        "p50_ttft_ms":40.,"p95_latency_ms":300.,
                        "output_tokens_per_second":100.*concurrency,
                        "requests_per_second":2.*concurrency,
                    },
                    "generated_tokens_mean":25. if mode=="short" else 128.,
                    "bucket_statistics":group,"requests":records,
                }
                (root/f"{mode}_c{concurrency}_seed{seed}.json").write_text(
                    json.dumps(record),encoding="utf-8"
                )


class Stage7AnalyzerContractTests(unittest.TestCase):
    def test_all_16_matched_cases_analyzed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fixture(root)
            d=analyze(root)
            self.assertEqual(d["status"],"pass")
            self.assertEqual(d["total_measured_requests"],384)
            self.assertEqual(len(d["cases"]),16)
            self.assertEqual(len(d["two_seed_observed_frontier"]["short"]),4)
            self.assertEqual(len(d["observed_slo_decisions"]["long"]),6)
            self.assertTrue(d["not_new_gpu_measurement"])

    def test_changed_passage_across_concurrencies_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            path=root/"short_c8_seed42.json"
            d=json.loads(path.read_text())
            d["requests"][0]["prompt_sha256"]="f"*64
            path.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(root)

    def test_generation_length_and_token_usage_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            p=root/"long_c4_seed137.json"
            d=json.loads(p.read_text())
            d["generated_tokens_mean"]=21.
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(root)
            d["generated_tokens_mean"]=128.
            d["summary"]["prompt_usage_coverage"]=0.
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(root)

    def test_missing_case_is_not_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            (root/"short_c2_seed42.json").unlink()
            with self.assertRaises(FileNotFoundError):
                analyze(root)


if __name__=="__main__":
    unittest.main()
