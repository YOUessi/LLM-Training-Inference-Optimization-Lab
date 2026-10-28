"""Real Stage11 held-out seed evidence; no synthetic GPU measurements.

Recompute the published analysis from the original 12 real vLLM SSE JSON
files. Every rejection counts against the full offered-request SLO.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.stage11_heldout_analyze import analyze

ROOT=Path(__file__).resolve().parents[1]/"docs/evidence/stage11_heldout"
SOURCE="e5356b799e9916fbda91f79ba99703c08f007b62"


def load(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))


class Stage11RealHeldoutEvidenceTests(unittest.TestCase):
    def test_source_commit_raw_sha_and_auto_job_provenance(self):
        m=load("stage11_manifest.json")
        self.assertEqual(m["schema"],"stage11-heldout-admission-gpu-evidence-v1")
        self.assertEqual(m["experiment_source_sha"],SOURCE)
        self.assertEqual(m["new_heldout_seeds"],[2027,3031])
        self.assertEqual(m["previously_tuned_seeds"],[42,137])
        self.assertEqual((m["offered"],m["completed"],m["rejected"],m["errors"]),
                         (288,278,10,0))
        self.assertTrue(m["cpu_analysis_completed_by_10minute_approved_auto_executor"])
        self.assertEqual(len(m["source_data_raw_cases_sha256"]),12)
        for name,sha in m["source_data_raw_cases_sha256"].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),sha)
        self.assertEqual(hashlib.sha256((ROOT/"stage11_analysis.json").read_bytes()).hexdigest(),
                         m["analysis_sha256"])
        s=load("auto_progress_state.json")
        self.assertEqual(s["source_sha"],SOURCE)
        for id in ("heldout_seed_2027","heldout_seed_3031","stage11_heldout_analysis"):
            self.assertEqual(s["jobs"][id]["status"],"passed")

    def test_288_offered_and_no_silent_drop(self):
        total=done=drop=0
        for seed in (2027,3031):
            for mode in ("label","detailed"):
                for policy,deadline in (("unlimited",250),("bounded",150),("bounded",400)):
                    f=f"stage11_seed_{seed}/{mode}_{policy}_d{deadline}_seed{seed}.json"
                    x=load(f)
                    with self.subTest(file=f):
                        self.assertEqual(x["status"],"measured")
                        self.assertEqual(x["environment"]["git_commit"],SOURCE)
                        s=x["summary"];cfg=x["config"]
                        self.assertEqual(s["offered"],24)
                        self.assertEqual(s["completed"]+s["rejected"],24)
                        self.assertEqual(s["errors"],0)
                        self.assertEqual(cfg["admission_deadline_ms"],deadline)
                        self.assertEqual(cfg["fixed_offered_rate"],8.)
                        self.assertEqual(x["accepted_output_length_gate"]["status"],"pass")
                        self.assertTrue(x["server_queue_monitor"]["available"])
                        self.assertEqual(len(x["requests"]),24)
                        for lang in ("en","zh-cn","ja"):
                            self.assertEqual(s["by_language"][lang]["offered"],8)
                        total+=s["offered"];done+=s["completed"];drop+=s["rejected"]
        self.assertEqual((total,done,drop),(288,278,10))

    def test_independent_recompute_and_policy_cost(self):
        frozen=load("stage11_analysis.json")
        self.assertEqual(frozen,analyze(ROOT))
        self.assertEqual(frozen["status"],"pass")
        self.assertEqual(frozen["offered"],288)
        self.assertEqual(len(frozen["two_seed_aggregate"]),6)
        rows={(x["mode"],x["policy"],x["deadline_ms"]):x
              for x in frozen["two_seed_aggregate"]}
        self.assertEqual(rows[("detailed","unlimited",250)]["completed"],48)
        self.assertEqual(rows[("detailed","bounded",150)]["completed"],42)
        self.assertEqual(rows[("detailed","bounded",400)]["completed"],44)
        self.assertEqual(rows[("detailed","bounded",150)]["rejected"],6)
        self.assertAlmostEqual(rows[("detailed","unlimited",250)]["compliance_250ms_fraction"],
                               35/48)
        self.assertAlmostEqual(rows[("detailed","bounded",150)]["compliance_250ms_fraction"],
                               42/48)
        self.assertAlmostEqual(rows[("detailed","bounded",400)]["compliance_250ms_fraction"],
                               39/48)
        for policy,deadline in (("unlimited",250),("bounded",150),("bounded",400)):
            self.assertEqual(rows[("label",policy,deadline)]["rejected"],0)
            self.assertEqual(rows[("label",policy,deadline)]["compliance_250ms_fraction"],1.)


if __name__=="__main__":
    unittest.main()
