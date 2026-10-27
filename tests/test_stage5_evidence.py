"""Stage 5 archival evidence and provenance contract (no GPU required here)."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


EVIDENCE=Path(__file__).resolve().parents[1]/"docs"/"evidence"/"stage5_kv_and_mixed"


def load(name):
    return json.loads((EVIDENCE/name).read_text(encoding="utf-8"))


class Stage5EvidenceTests(unittest.TestCase):
    def test_capacity_retention_and_eviction_gates(self):
        p=load("stage5_eviction_20261028/capacity.json")
        self.assertEqual(p["status"],"pass")
        self.assertEqual(p["kind"],"vllm_prefix_cache_capacity_eviction_probe")
        self.assertEqual(p["config"]["num_gpu_blocks_override"],256)
        self.assertEqual(p["config"]["distinct_fillers"],12)
        self.assertEqual(len(p["events"]),16)
        self.assertTrue(all(p["gates"].values()))
        key={r["stage"]:r for r in p["events"]}
        self.assertLessEqual(key["initial_miss"]["cache"]["cache_hit_fraction"],.10)
        self.assertGreaterEqual(key["immediate_replay"]["cache"]["cache_hit_fraction"],.75)
        self.assertLessEqual(key["post_pressure_replay"]["cache"]["cache_hit_fraction"],.25)
        self.assertGreaterEqual(key["restored_replay"]["cache"]["cache_hit_fraction"],.75)
        self.assertTrue(all(250 <= r["prompt_tokens"] <= 1700 for r in p["events"]))

    def test_full_192_mixed_real_model_requests(self):
        results=load("stage5_summary.json")
        self.assertEqual(results["status"],"pass")
        self.assertEqual(results["mixed"]["total_completed_requests"],192)
        self.assertEqual(len(results["mixed"]["by_case"]),8)
        for concurrency in (1,2,4,8):
            for seed in (42,137):
                p=load(f"stage5_mixed_20261028/mixed_c{concurrency}_seed{seed}.json")
                with self.subTest(concurrency=concurrency,seed=seed):
                    self.assertEqual(p["status"],"measured")
                    self.assertEqual(p["kind"],"vllm_mixed_context_real_generation_workload")
                    self.assertEqual(p["config"]["concurrency"],concurrency)
                    self.assertEqual(p["config"]["seed"],seed)
                    self.assertEqual(p["summary"]["succeeded"],24)
                    self.assertEqual(p["summary"]["failed"],0)
                    self.assertEqual(p["summary"]["prompt_usage_coverage"],1.)
                    self.assertEqual(p["summary"]["usage_coverage"],1.)
                    self.assertEqual(len(p["requests"]),24)
                    self.assertEqual(set(p["by_prompt_length"]),
                                     {"short","medium","long","very_long"})
                    for bucket in p["by_prompt_length"].values():
                        self.assertEqual(bucket["status"],"pass")
                        self.assertEqual(bucket["success"],6)
                        self.assertEqual(bucket["sent"],6)
                        self.assertGreater(bucket["p95_ttft_ms"],0)

    def test_manifest_raw_sha256_and_actual_server_flags(self):
        m=load("stage5_manifest.json")
        self.assertEqual(m["schema"],"stage5-kv-and-mixed-context-evidence-v1")
        self.assertEqual(m["experiment_source_commit"],
                         "e7c96b8f9568885074630f4727e274a31cb862ed")
        self.assertTrue(m["capacity_cache_enabled_log_verified"])
        self.assertTrue(m["mixed_cache_disabled_log_verified"])
        self.assertEqual(m["num_blocks_override_eviction"],256)
        self.assertEqual(len(m["source_files"]),9)
        for item in m["source_files"]:
            with self.subTest(file=item["relative"]):
                data=(EVIDENCE/item["relative"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(),item["sha256"])

    def test_reports_genuinely_published(self):
        docs=EVIDENCE.parents[1]
        self.assertTrue((docs/"STAGE5_KV_CAPACITY_REPORT_2026-10-28.md").exists())
        self.assertTrue((docs/"STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md").exists())


if __name__=="__main__":
    unittest.main()
