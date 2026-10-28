"""CPU gates for the read-only 10-minute audit and pinned Tang GPU queue."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.auto_progress_tick import JOBS, SCHEMA, tick, verify_completed_job
from scripts.stage11_cpu_preflight import preflight
from scripts.stage11_heldout_analyze import analyze


ROOT=Path(__file__).resolve().parents[1]


class Stage11AutomationTests(unittest.TestCase):
    def test_cpu_preflight_has_approved_finite_plan(self):
        data=preflight(ROOT)
        self.assertEqual(data["status"],"preflight_pass")
        self.assertEqual(data["expected_offered_requests"],288)
        self.assertEqual(data["heldout_seed_set"],[2027,3031])
        self.assertEqual(data["policy_deadlines_ms"],[150,400])
        self.assertFalse(data["gpu_access_from_github_actions"])
        self.assertEqual([x["id"] for x in data["approved_work_items"]],
                         [x["id"] for x in JOBS[:3]])

    def test_no_unauthorized_gpu_job_or_remote_code_pull(self):
        self.assertEqual(len(JOBS),11)
        for job in JOBS:
            self.assertTrue(job["id"].startswith(("heldout_seed_","stage11_heldout_analysis",
                                                   "stage12_seed_","stage12_crossdomain_analysis",
                                                   "stage13_k8s_","stage13_prose_","stage13_all_analysis")))
            self.assertNotIn("git",job["argv"])
            self.assertNotIn("curl",job["argv"])
            self.assertNotIn("pip",job["argv"])
        self.assertEqual(SCHEMA,"llm-opt-tang-autoprogress-v1")

    def test_missing_pinned_approval_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaisesRegex(RuntimeError,"APPROVAL_REQUIRED"):
                tick(root,dry_run=True)

    def test_heldout_analysis_requires_real_seed_data(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                analyze(Path(temp))

    def test_new_policies_cannot_be_claimed_without_results(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                verify_completed_job(Path(temp),"heldout_seed_2027")
            with self.assertRaises(FileNotFoundError):
                verify_completed_job(Path(temp),"stage11_heldout_analysis")


if __name__=="__main__":
    unittest.main()
