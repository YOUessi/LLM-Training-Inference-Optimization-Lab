"""Honest archival validation: 288 real Poisson arrivals with unequal realized rates."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]/"docs/evidence/stage8_unconditioned"


class Stage8PoissonEvidenceTests(unittest.TestCase):
    def test_real_requests_and_original_hashes(self):
        manifest=json.loads((ROOT/"unconditioned_manifest.json").read_text())
        self.assertEqual(manifest["measured_requests"],288)
        self.assertEqual(len(manifest["raw_files"]),12)
        self.assertEqual(manifest["source_git_commit"],
                         "8a6053a1d4ee8fa3850dd362a9ebb6ab9039b7f8")
        for fname,digest in manifest["raw_files"].items():
            p=ROOT/"unconditioned"/fname
            with self.subTest(name=fname):
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),digest)
                d=json.loads(p.read_text())
                self.assertEqual(d["status"],"measured")
                self.assertEqual(d["summary"]["succeeded"],24)
                self.assertEqual(d["summary"]["failed"],0)
                self.assertEqual(d["summary"]["prompt_usage_coverage"],1.)
                self.assertEqual(d["summary"]["usage_coverage"],1.)
                self.assertTrue(d["all_actual_input_tokens_in_bounds"])
                self.assertEqual(d["length_gates"]["status"],"pass")

    def test_unconditioned_rate_variance_remains_disclosed(self):
        d=json.loads((ROOT/"unconditioned_analysis.json").read_text())
        self.assertEqual(d["status"],"pass")
        self.assertEqual(d["measured_requests"],288)
        rates={round(x["empirical_offered_rate"],2)
               for x in d["per_case"]}
        self.assertEqual(rates,{1.48,2.86,5.93,11.45})
        self.assertTrue(any(x["p95_arrival_to_ttft_ms"]>900 for x in d["per_case"]))
        self.assertEqual(len(d["per_case"]),12)


if __name__=="__main__":
    unittest.main()
