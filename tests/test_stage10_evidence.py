"""Check Stage10 real GPU offers/rejections, SHA256 and all-offered SLO."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage10_analysis import analyze

ROOT=Path(__file__).resolve().parents[1]/"docs/evidence/stage10_admission"


def load(relative):
    return json.loads((ROOT/relative).read_text(encoding="utf-8"))


class Stage10EvidenceTests(unittest.TestCase):
    def test_complete_raw_provenance(self):
        m=load("stage10_manifest.json")
        self.assertEqual(m["schema"],"stage10-real-client-admission-evidence-v1")
        self.assertEqual(m["experiment_source_commit"],
                         "1f654f7404a7cc7e1cd5a88f28fc0615349a154f")
        self.assertEqual(m["analysis_source_commit"],
                         "7b4618b5f1a1d66fe3168d69763bde77ac0856af")
        self.assertEqual(m["offered"],192)
        self.assertEqual(m["completed"],186)
        self.assertEqual(m["rejected"],6)
        self.assertEqual(m["errors"],0)
        self.assertTrue(m["server_log_private_only_on_Tang"])
        self.assertTrue(m["cache_disabled_actual_server_log"])
        self.assertEqual(len(m["raw_files_sha256"]),8)
        for name,sha in m["raw_files_sha256"].items():
            with self.subTest(name=name):
                self.assertEqual(
                    hashlib.sha256((ROOT/"real_sse"/name).read_bytes()).hexdigest(),
                    sha
                )
        self.assertEqual(
            hashlib.sha256((ROOT/"stage10_analysis.json").read_bytes()).hexdigest(),
            m["analysis_sha256"]
        )

    def test_complete_offered_denominator_and_no_errors(self):
        total=completed=rejected=0
        for seed in (42,137):
            for policy in ("unlimited","bounded"):
                for mode in ("label","detailed"):
                    name=f"real_sse/{policy}_{mode}_seed{seed}.json"
                    d=load(name)
                    with self.subTest(name=name):
                        self.assertEqual(d["status"],"measured")
                        self.assertEqual(d["kind"],
                                         "stage10_real_client_admission_policy_study")
                        self.assertEqual(d["summary"]["offered"],24)
                        self.assertEqual(d["summary"]["errors"],0)
                        self.assertEqual(d["summary"]["accepted"]+
                                         d["summary"]["rejected"],24)
                        self.assertEqual(d["summary"]["completed"]+
                                         d["summary"]["rejected"],24)
                        self.assertEqual(d["accepted_output_length_gate"]["status"],"pass")
                        self.assertTrue(d["server_queue_monitor"]["available"])
                        self.assertTrue(d["gpu_kv_monitor"]["available"])
                        self.assertEqual(len(d["requests"]),24)
                        for x in d["requests"]:
                            self.assertIn(x["status"],("ok","rejected"))
                            if x["status"]=="rejected":
                                self.assertIn(x["rejection_reason"],
                                              ("queue_full","deadline_expired"))
                                self.assertIsNone(x["ttft_from_due_ms"])
                        total+=24
                        completed+=d["summary"]["completed"]
                        rejected+=d["summary"]["rejected"]
        self.assertEqual((total,completed,rejected),(192,186,6))

    def test_policy_result_preserves_rejected_traffic_penalty(self):
        a=load("stage10_analysis.json")
        self.assertEqual(a,analyze(ROOT/"real_sse"))
        self.assertEqual(a["total_offered_requests"],192)
        self.assertEqual(len(a["paired_cases"]),4)
        heavy=[x for x in a["paired_cases"] if x["mode"]=="detailed"]
        self.assertEqual(len(heavy),2)
        self.assertEqual(sum(x["unlimited"]["completed"] for x in heavy),48)
        self.assertEqual(sum(x["bounded"]["completed"] for x in heavy),42)
        for x in heavy:
            self.assertEqual(x["bounded"]["rejected"],3)
            self.assertIsNone(x["bounded"]["all_offered_p95_ttft_ms"])
            self.assertGreater(x["delta_all_offered_250ms_slo_fraction"],0)
            self.assertLess(x["delta_useful_tokens_per_second"],0)
        controls=[x for x in a["paired_cases"] if x["mode"]=="label"]
        self.assertTrue(all(x["unlimited"]["rejected"]==0 and
                            x["bounded"]["rejected"]==0 for x in controls))


if __name__=="__main__":
    unittest.main()
