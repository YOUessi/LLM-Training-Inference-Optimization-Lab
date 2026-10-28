"""Stage13 finite pinned research queue and CPU manifest tests."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.auto_progress_tick import JOBS,verify_completed_job
from scripts.stage13_cpu_preflight import preflight

ROOT=Path(__file__).resolve().parents[1]


class Stage13FiniteExecutionTests(unittest.TestCase):
    def test_preregistered_768_offers_and_two_domain_sources(self):
        d=preflight(ROOT)
        self.assertEqual(d["status"],"pass")
        self.assertEqual(d["all_offered"],768)
        self.assertEqual(d["real_gpu_cases"],32)
        self.assertEqual(d["offered_rate_per_second"],[8.,16.])
        self.assertEqual(len(d["source_domains"]),2)
        self.assertEqual(len(d["approved_gpu_jobs"]),4)
        self.assertEqual(d["approved_cpu_job"],"stage13_all_analysis")
        self.assertFalse(d["github_hosted_has_tang_gpu"])
        self.assertFalse(d["changes_vllm_internal_scheduler"])

    def test_approved_tasks_finite_no_mutable_code_download(self):
        self.assertEqual(len(JOBS),11)
        self.assertEqual([j["id"] for j in JOBS[6:]],
                         ["stage13_k8s_seed6101","stage13_k8s_seed6109",
                          "stage13_prose_seed7201","stage13_prose_seed7209",
                          "stage13_all_analysis"])
        self.assertTrue(all(j["requires_gpu"] for j in JOBS[6:10]))
        self.assertFalse(JOBS[10]["requires_gpu"])
        for job in JOBS[6:]:
            self.assertNotIn("git",job["argv"])
            self.assertNotIn("curl",job["argv"])
            self.assertNotIn("pip",job["argv"])

    def test_cron_bare_python_loader_is_local_only_and_not_module_missing(self):
        import os
        import subprocess
        import sys
        env=os.environ.copy()
        env.pop("PYTHONPATH",None)
        code=(
            "from scripts.auto_progress_tick import verify_completed_job;"
            "from pathlib import Path;"
            "verify_completed_job(Path.cwd(), 'stage13_k8s_seed6101')"
        )
        completed=subprocess.run(
            [sys.executable,"-c",code],cwd=ROOT,
            env=env,text=True,capture_output=True,timeout=10,check=False
        )
        # No real data exists in local Mac results, so FileNotFoundError
        # is correct. The old cron bug was ModuleNotFoundError.
        self.assertNotEqual(completed.returncode,0)
        self.assertIn("FileNotFoundError",completed.stderr)
        self.assertNotIn("ModuleNotFoundError",completed.stderr)

    def test_without_real_gpu_results_job_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError):
                verify_completed_job(root,"stage13_k8s_seed6101")
            with self.assertRaises(FileNotFoundError):
                verify_completed_job(root,"stage13_prose_seed7209")
            with self.assertRaises(FileNotFoundError):
                verify_completed_job(root,"stage13_all_analysis")


if __name__ == "__main__":
    unittest.main()
