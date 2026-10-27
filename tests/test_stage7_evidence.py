"""Validate complete Stage7 archived real GPU evidence and rejected first-run input gate."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage7_analysis import analyze


ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"docs/evidence/stage7_natural"


def read(relative: str):
    return json.loads((EVIDENCE/relative).read_text(encoding="utf-8"))


class Stage7RealGpuEvidenceTests(unittest.TestCase):
    def test_verified_source_hash_and_count(self):
        meta=read("stage7_source_manifest.json")
        self.assertEqual(meta["schema"],"stage7-natural-nonrepeating-raw-evidence-v1")
        self.assertEqual(meta["experiment_git_sha"],"00993045f8564539428ce65c5a9170f5c840ab1d")
        self.assertEqual(meta["analysis_git_sha"],"38d32600b7f6e74f08b290bd11a198f7bf6e5f34")
        corpus=ROOT/"data/stage7_public_domain_passages.jsonl"
        self.assertEqual(hashlib.sha256(corpus.read_bytes()).hexdigest(),meta["corpus_jsonl_sha256"])
        self.assertEqual(len(meta["raw_artifact_hashes"]),22)
        for rel,digest in meta["raw_artifact_hashes"].items():
            with self.subTest(rel=rel):
                self.assertEqual(hashlib.sha256((EVIDENCE/rel).read_bytes()).hexdigest(),digest)
        self.assertTrue(meta["actual_prefix_cache_disabled_in_server_log"])
        self.assertTrue(meta["server_log_not_committed_to_github"])

    def test_384_success_and_expected_generation_lengths(self):
        count=0
        for seed in (42,137):
            for mode in ("short","long"):
                for concurrency in (1,2,4,8):
                    name=f"stage7_corrected/{mode}_c{concurrency}_seed{seed}.json"
                    d=read(name)
                    with self.subTest(name=name):
                        self.assertEqual(d["status"],"measured")
                        self.assertEqual(d["environment"]["git_commit"],"00993045f8564539428ce65c5a9170f5c840ab1d")
                        self.assertEqual(d["summary"]["succeeded"],24)
                        self.assertEqual(d["summary"]["failed"],0)
                        self.assertEqual(d["summary"]["prompt_usage_coverage"],1.)
                        self.assertEqual(d["summary"]["usage_coverage"],1.)
                        self.assertEqual(len(d["requests"]),24)
                        self.assertEqual(len({x["doc_id"] for x in d["requests"]}),24)
                        self.assertEqual(len({x["genre"] for x in d["requests"]}),5)
                        self.assertTrue(all(x["status"]=="pass" and x["sent"]==6 for x in d["bucket_statistics"].values()))
                        self.assertGreater(d["summary"]["p95_ttft_ms"],0)
                        self.assertGreater(d["summary"]["output_tokens_per_second"],0)
                        if mode=="long":
                            self.assertEqual(d["generated_tokens_mean"],128.)
                        else:
                            self.assertTrue(25 <= d["generated_tokens_mean"] <= 32)
                        count+=24
        self.assertEqual(count,384)

    def test_old_invalid_run_kept_and_not_counted(self):
        legacy=read("stage7_initial_invalid_summary.json")
        self.assertEqual(legacy["status"],"invalid_long_input_gate")
        self.assertEqual(legacy["first_attempt_real_requests"],120)
        self.assertFalse(legacy["accepted_into_corrected_384_request_matrix"])
        invalid=read("stage7_invalid_first_run/long_c1_seed42.json")
        self.assertEqual(invalid["status"],"invalid_or_blocked")
        self.assertEqual(invalid["bucket_statistics"]["short"]["status"],"invalid")
        self.assertEqual(invalid["bucket_statistics"]["short"]["max_actual_prompt_tokens"],269)
        self.assertEqual(invalid["generated_tokens_mean"],128.)

    def test_machine_analysis_recomputes_from_archived_raw(self):
        raw=read("stage7_analysis.json")
        recomputed=analyze(EVIDENCE/"stage7_corrected")
        self.assertEqual(raw,recomputed)
        self.assertEqual(raw["status"],"pass")
        self.assertEqual(raw["total_measured_requests"],384)
        self.assertEqual(raw["case_count"],16)
        self.assertTrue(raw["not_new_gpu_measurement"])
        self.assertEqual(set(raw["two_seed_observed_frontier"]),{"short","long"})


if __name__=="__main__":
    unittest.main()
