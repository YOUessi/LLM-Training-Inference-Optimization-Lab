"""Verified independent-domain real GPU admission evidence (no fabricated results)."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from scripts.stage12_crossdomain_analyze import analyze

ROOT=Path(__file__).resolve().parents[1]/"docs/evidence/stage12_crossdomain"
SHA="385cc86535781f5dec92fbd5d59faa4e5bc47a36"


def load(path:str):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


class Stage12RealCrossdomainEvidenceTests(unittest.TestCase):
    def test_original_runtime_flags_and_source_hashes(self):
        manifest=load("stage12_manifest.json")
        self.assertEqual(manifest["schema"],"stage12-crossdomain-admission-real-gpu-evidence-v1")
        self.assertEqual(manifest["experiment_git_sha"],SHA)
        self.assertEqual(manifest["request_frozen_sha256"],
                         "fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04")
        self.assertEqual((manifest["offered"],manifest["completed"],
                          manifest["explicit_rejected"],manifest["unexpected_errors"]),
                         (288,278,10,0))
        self.assertEqual(len(manifest["raw_file_sha256"]),12)
        self.assertTrue(manifest["server_logs_remain_private_on_Tang"])
        for name,expected in manifest["raw_file_sha256"].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                                 expected)
        self.assertEqual(hashlib.sha256((ROOT/"stage12_analysis.json").read_bytes()).hexdigest(),
                         manifest["analysis_sha256"])

    def test_288_offered_and_semantics_all_cases(self):
        offered=completed=rejected=0
        for seed in (4093,5087):
            for mode in ("label","detailed"):
                for policy,deadline in (("unlimited",250),("bounded",150),("bounded",400)):
                    name=f"stage12_seed_{seed}/{mode}_{policy}_d{deadline}_seed{seed}.json"
                    r=load(name)
                    with self.subTest(case=name):
                        self.assertEqual(r["status"],"measured")
                        self.assertEqual(r["environment"]["git_commit"],SHA)
                        self.assertEqual(r["config"]["frozen_requests_sha256"],
                                         "fa338ac75be4795326663b9935fcd1cebbb4e2f1d15ffaa69d6b7a604a0afd04")
                        self.assertEqual(r["config"]["seed"],seed)
                        self.assertEqual(r["config"]["policy"],policy)
                        self.assertEqual(r["config"]["admission_deadline_ms"],deadline)
                        s=r["summary"]
                        self.assertEqual(s["offered"],24)
                        self.assertEqual(s["completed"]+s["rejected"],24)
                        self.assertEqual(s["errors"],0)
                        self.assertEqual(len(r["requests"]),24)
                        self.assertEqual(s["by_language"]["en"]["offered"],24)
                        self.assertEqual(s["by_language"]["ja"]["offered"],0)
                        self.assertEqual(s["by_language"]["zh-cn"]["offered"],0)
                        self.assertEqual(r["accepted_output_length_gate"]["status"],"pass")
                        self.assertTrue(r["server_queue_monitor"]["available"])
                        self.assertTrue(r["gpu_kv_monitor"]["available"])
                        if s["rejected"]:
                            self.assertIsNone(s["p95_all_offered_ttft_ms"])
                        offered+=24;completed+=s["completed"];rejected+=s["rejected"]
        self.assertEqual((offered,completed,rejected),(288,278,10))

    def test_independent_analysis_exact_recompute(self):
        evidence=load("stage12_analysis.json")
        self.assertEqual(evidence,analyze(ROOT))
        self.assertEqual(evidence["source_git_commit"],SHA)
        self.assertEqual(evidence["status"],"pass")
        self.assertEqual(evidence["offered"],288)
        self.assertEqual(evidence["recorded_12_cases"],12)
        self.assertEqual(set(evidence["heldout_seeds"]),{4093,5087})
        agg={(p["mode"],p["policy"],p["deadline_ms"]):p
             for p in evidence["two_seed_aggregate"]}
        self.assertEqual(agg[("detailed","unlimited",250)]["compliant_250ms"],35)
        self.assertEqual(agg[("detailed","bounded",150)]["compliant_250ms"],41)
        self.assertEqual(agg[("detailed","bounded",400)]["compliant_250ms"],42)
        self.assertEqual(agg[("detailed","bounded",150)]["rejected"],7)
        self.assertEqual(agg[("detailed","bounded",400)]["rejected"],3)
        self.assertEqual(agg[("detailed","unlimited",250)]["rejected"],0)

    def test_finished_reviewed_automation_no_running_gpu_jobs(self):
        state=load("stage12_auto_progress_state.json")
        self.assertEqual(state["source_sha"],SHA)
        for key in ("heldout_seed_2027","heldout_seed_3031",
                    "stage11_heldout_analysis","stage12_seed_4093",
                    "stage12_seed_5087","stage12_crossdomain_analysis"):
            self.assertEqual(state["jobs"][key]["status"],"passed")


if __name__=="__main__":
    unittest.main()
