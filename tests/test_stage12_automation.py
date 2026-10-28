"""Finite approved Stage12 cross-domain work and source-integrity contracts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.auto_progress_tick import JOBS,verify_completed_job
from scripts.stage12_cpu_preflight import preflight
from scripts.stage12_crossdomain_analyze import analyze


ROOT=Path(__file__).resolve().parents[1]


class Stage12AutomationContracts(unittest.TestCase):
    def test_corpus_and_approved_job_identity(self):
        d=preflight(ROOT)
        self.assertEqual(d["status"],"pass")
        self.assertEqual(d["source_document_count"],24)
        self.assertEqual(d["genres"],5)
        self.assertEqual(d["approved_new_seeds"],[4093,5087])
        self.assertEqual(d["expected_stage12_offered"],288)
        self.assertEqual(d["planned_jobs"],[x["id"] for x in JOBS[3:]])
        self.assertFalse(d["git_hosted_runner_gpu_access"])

    def test_gpu_queue_is_finite_and_no_fetch(self):
        self.assertEqual(len(JOBS),6)
        for j in JOBS[3:]:
            self.assertNotIn("git",j["argv"])
            self.assertNotIn("curl",j["argv"])
            self.assertNotIn("pip",j["argv"])
        self.assertEqual(JOBS[-1]["id"],"stage12_crossdomain_analysis")
        self.assertFalse(JOBS[-1]["requires_gpu"])

    def test_cannot_claim_experiment_without_actual_model_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError):
                analyze(root)
            with self.assertRaises(FileNotFoundError):
                verify_completed_job(root,"stage12_seed_4093")
            with self.assertRaises(FileNotFoundError):
                verify_completed_job(root,"stage12_crossdomain_analysis")


if __name__=="__main__":
    unittest.main()
