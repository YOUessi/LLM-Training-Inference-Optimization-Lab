"""Roadmap audit is read-only and reports verifiable next steps only."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plan_watch import audit, main


class PlanWatchTests(unittest.TestCase):
    def test_repo_audit_is_read_only_and_stage5_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"docs").mkdir()
            (root/"docs/ROADMAP.md").write_text(
                "## Stage 5\n- [x] code\n- [ ] GPU data\n")
            (root/"src/llm_opt_lab/serving").mkdir(parents=True)
            d=audit(root)
            self.assertEqual(d["status"],"stage5_evidence_pending")
            self.assertEqual(d["roadmap_checkbox_counts"],
                             {"completed":1,"open":1})
            self.assertFalse(d["gpu_access_available_in_hosted_ci"])
            self.assertTrue(any("Tang" in x for x in d["next_executable_actions"]))

    def test_next_action_advances_when_evidence_reaches_stage6(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"docs").mkdir()
            (root/"docs/ROADMAP.md").write_text("- [x] Stage5\n")
            for report in ("STAGE5_KV_CAPACITY_REPORT_2026-10-28.md",
                           "STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md",
                           "STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md"):
                (root/"docs"/report).write_text("measured evidence")
            data=audit(root)
            self.assertEqual(data["status"],"stage5_evidence_published")
            self.assertTrue(data["published_reports"]["Stage 6 SLO frontier"])
            self.assertTrue(any("128/256/512" in action
                                for action in data["next_executable_actions"]))

    def test_multi_capacity_completion_advances_to_diverse_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"docs").mkdir()
            (root/"docs/ROADMAP.md").write_text("- [x] stage6\n")
            for name in ("STAGE5_KV_CAPACITY_REPORT_2026-10-28.md",
                         "STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md",
                         "STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md",
                         "STAGE6_CAPACITY_MATRIX_REPORT_2026-10-28.md"):
                (root/"docs"/name).write_text("measured")
            d=audit(root)
            self.assertTrue(d["published_reports"]["Stage 6 KV capacity matrix"])
            self.assertTrue(any("nonrepetitive corpus" in action
                                for action in d["next_executable_actions"]))

    def test_stage7_report_advances_watch_to_stage8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"docs").mkdir()
            (root/"docs/ROADMAP.md").write_text("- [x] stage7\n")
            for name in ("STAGE5_KV_CAPACITY_REPORT_2026-10-28.md",
                         "STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md",
                         "STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md",
                         "STAGE6_CAPACITY_MATRIX_REPORT_2026-10-28.md",
                         "STAGE7_REPORT_2026-10-28.md"):
                (root/"docs"/name).write_text("real measured")
            d=audit(root)
            self.assertEqual(d["status"],"stage7_evidence_published")
            self.assertTrue(d["published_reports"]["Stage 7 diverse natural prose"])
            self.assertTrue(any("Stage8" in action for action in d["next_executable_actions"]))

    def test_stage8_completed_advances_to_stage9_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"docs").mkdir()
            (root/"docs/ROADMAP.md").write_text("- [x] stage8\\n")
            for name in ("STAGE5_KV_CAPACITY_REPORT_2026-10-28.md",
                         "STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md",
                         "STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md",
                         "STAGE6_CAPACITY_MATRIX_REPORT_2026-10-28.md",
                         "STAGE7_REPORT_2026-10-28.md",
                         "STAGE8_REPORT_2026-10-28.md"):
                (root/"docs"/name).write_text("measured source")
            d=audit(root)
            self.assertEqual(d["status"],"stage8_evidence_published")
            self.assertTrue(d["published_reports"]["Stage 8 matched-rate serving"])
            self.assertTrue(any("Stage9" in action for action in d["next_executable_actions"]))
            self.assertTrue(any("client queue" in action for action in d["next_executable_actions"]))

    def test_stage9_queue_work_published_advances_to_honest_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"docs").mkdir()
            (root/"docs/ROADMAP.md").write_text("- [x] stage9\\n")
            for name in ("STAGE5_KV_CAPACITY_REPORT_2026-10-28.md",
                         "STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md",
                         "STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md",
                         "STAGE6_CAPACITY_MATRIX_REPORT_2026-10-28.md",
                         "STAGE7_REPORT_2026-10-28.md",
                         "STAGE8_REPORT_2026-10-28.md",
                         "STAGE9_REPORT_2026-10-28.md"):
                (root/"docs"/name).write_text("real measured")
            d=audit(root)
            self.assertEqual(d["status"],"stage9_queue_baseline_published")
            self.assertTrue(d["published_reports"]["Stage 9 worker queue isolation"])
            self.assertTrue(any("Stage10" in item for item in d["next_executable_actions"]))
            self.assertTrue(any("rejected" in item for item in d["next_executable_actions"]))

    def test_stage10_admission_result_advances_to_heldout_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"docs").mkdir()
            (root/"docs/ROADMAP.md").write_text("- [x] stage10\\n")
            for name in ("STAGE5_KV_CAPACITY_REPORT_2026-10-28.md",
                         "STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md",
                         "STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md",
                         "STAGE6_CAPACITY_MATRIX_REPORT_2026-10-28.md",
                         "STAGE7_REPORT_2026-10-28.md",
                         "STAGE8_REPORT_2026-10-28.md",
                         "STAGE9_REPORT_2026-10-28.md",
                         "STAGE10_REPORT_2026-10-28.md"):
                (root/"docs"/name).write_text("measured")
            d=audit(root)
            self.assertEqual(d["status"],"stage10_admission_ablation_published")
            self.assertTrue(d["published_reports"]["Stage 10 client admission A/B"])
            self.assertTrue(any("Stage11" in action for action in d["next_executable_actions"]))
            self.assertTrue(any("rejected" in action for action in d["next_executable_actions"]))

    def test_stage11_heldout_published_advances_to_independent_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"docs").mkdir()
            (root/"docs/ROADMAP.md").write_text("- [x] stage11\\n")
            for name in ("STAGE5_KV_CAPACITY_REPORT_2026-10-28.md",
                         "STAGE5_MIXED_CONTEXT_REPORT_2026-10-28.md",
                         "STAGE6_SLO_FRONTIER_REPORT_2026-10-28.md",
                         "STAGE6_CAPACITY_MATRIX_REPORT_2026-10-28.md",
                         "STAGE7_REPORT_2026-10-28.md",
                         "STAGE8_REPORT_2026-10-28.md",
                         "STAGE9_REPORT_2026-10-28.md",
                         "STAGE10_REPORT_2026-10-28.md",
                         "STAGE11_REPORT_2026-10-28.md"):
                (root/"docs"/name).write_text("published measured evidence")
            d=audit(root)
            self.assertEqual(d["status"],"stage11_heldout_evidence_published")
            self.assertTrue(d["published_reports"]["Stage 11 held-out report"])
            self.assertTrue(any("Stage12" in action for action in d["next_executable_actions"]))
            self.assertTrue(any("queue is exhausted" in action
                                for action in d["next_executable_actions"]))

    def test_watch_summary_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"docs").mkdir()
            (root/"docs/ROADMAP.md").write_text("- [ ] Stage 5\n")
            out=root/"results/plan.json"
            summary=root/"results/summary.md"
            self.assertEqual(main(["--repo-root",str(root),
                                   "--out",str(out),"--summary",str(summary)]),0)
            p=json.loads(out.read_text())
            self.assertEqual(p["roadmap_checkbox_counts"]["open"],1)
            self.assertIn("GitHub hosted runners are CPU-only",summary.read_text())


if __name__=="__main__":
    unittest.main()
