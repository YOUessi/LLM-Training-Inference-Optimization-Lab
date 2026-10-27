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
