"""Validate Stage9 authentic SSE raw artifacts and reproducible queue analysis."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage9_worker_analysis import analyze

ROOT=Path(__file__).resolve().parents[1]/"docs/evidence/stage9_worker_isolation"
SHA="9f497193c38c8d2bb00b6b78fd25d6b07bc66528"


def load(filename: str):
    return json.loads((ROOT/filename).read_text(encoding="utf-8"))


class Stage9EvidenceTests(unittest.TestCase):
    def test_source_and_raw_checksum(self):
        meta=load("stage9_manifest.json")
        self.assertEqual(meta["schema"],"stage9-client-vs-server-queue-evidence-v1")
        self.assertEqual(meta["source_commit"],SHA)
        self.assertEqual(meta["offered_requests"],288)
        self.assertEqual(meta["successful_requests"],288)
        self.assertTrue(meta["server_cache_disabled_verified"])
        self.assertTrue(meta["server_logs_retained_only_on_Tang"])
        self.assertEqual(meta["server_max_num_seqs"],8)
        self.assertEqual(len(meta["raw_case_sha256"]),12)
        for name,digest in meta["raw_case_sha256"].items():
            with self.subTest(name=name):
                self.assertEqual(
                    hashlib.sha256((ROOT/"real_sse"/name).read_bytes()).hexdigest(),
                    digest,
                )
        self.assertEqual(
            hashlib.sha256((ROOT/"stage9_analysis.json").read_bytes()).hexdigest(),
            meta["analysis_sha256"],
        )

    def test_every_request_and_server_waiting_gauge(self):
        total=0
        for seed in (42,137):
            for workers in (8,32):
                for mode,rate in (("label",8),("detailed",2),("detailed",8)):
                    filename=f"real_sse/{mode}_w{workers}_rate{rate}_seed{seed}.json"
                    d=load(filename)
                    with self.subTest(filename=filename):
                        self.assertEqual(d["status"],"measured")
                        self.assertEqual(d["environment"]["git_commit"],SHA)
                        self.assertEqual(d["summary"]["succeeded"],24)
                        self.assertEqual(d["summary"]["failed"],0)
                        self.assertEqual(d["config"]["max_client_workers"],workers)
                        self.assertEqual(d["config"]["schedule_kind"],"fixed_window")
                        self.assertAlmostEqual(d["offered_arrival_rate_empirical_per_second"],rate)
                        self.assertTrue(d["all_actual_input_tokens_in_bounds"])
                        self.assertEqual(d["summary"]["prompt_usage_coverage"],1.)
                        self.assertEqual(d["summary"]["usage_coverage"],1.)
                        self.assertEqual(d["length_gates"]["status"],"pass")
                        q=d["server_request_queue_gauges"]
                        self.assertTrue(q["available"])
                        self.assertGreater(q["samples"],0)
                        self.assertGreaterEqual(q["max_server_waiting_requests"],0)
                        self.assertEqual(len(d["requests"]),24)
                        total+=24
        self.assertEqual(total,288)

    def test_recompute_matches_original_exactly(self):
        frozen=load("stage9_analysis.json")
        recomputed=analyze(ROOT/"real_sse")
        self.assertEqual(frozen,recomputed)
        self.assertEqual(frozen["status"],"pass")
        self.assertEqual(frozen["real_requests"],288)
        self.assertEqual(len(frozen["matched_pairs"]),6)

    def test_heavy_load_queue_migration_not_speedup(self):
        report=load("stage9_analysis.json")
        heavy=[x for x in report["matched_pairs"]
               if x["mode"]=="detailed" and x["target_offered_rate"]==8]
        self.assertEqual(len(heavy),2)
        self.assertTrue(all(x["p95_client_wait_reduction_ms"]>350 for x in heavy))
        self.assertTrue(all(x["p95_arrival_ttft_reduction_ms"]<30 for x in heavy))
        self.assertTrue(all(x["p95_dispatch_ttft_change_ms"]>300 for x in heavy))
        self.assertTrue(all(x["server_max_waiting_change"]>=3 for x in heavy))


if __name__=="__main__":
    unittest.main()
