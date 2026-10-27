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
