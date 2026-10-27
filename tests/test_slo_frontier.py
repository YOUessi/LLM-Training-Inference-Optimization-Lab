"""Stage 6 SLO frontier using the pinned Stage5 machine-measured evidence."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_opt_lab.serving.slo_frontier import analyze


EVIDENCE=Path(__file__).resolve().parents[1]/"docs"/"evidence"/"stage5_kv_and_mixed"/"stage5_mixed_20261028"


class SloFrontierTests(unittest.TestCase):
    def test_reproduce_real_stage5_frontier(self):
        d=analyze(EVIDENCE)
        self.assertTrue(d["not_new_gpu_experiment"])
        self.assertEqual(len(d["frontier_points"]),4)
        self.assertEqual(len(d["runs"]),8)
        expected={100.:1,150.:2,250.:4,500.:8}
        for x in d["slo_decisions"]:
            self.assertEqual(x["selected_concurrency"],
                             expected[x["p95_ttft_slo_ms"]])
            self.assertGreater(x["selected_conservative_output_tps"],0)
            self.assertLessEqual(x["actual_worst_p95_ttft_ms"],
                                 x["p95_ttft_slo_ms"])

    def test_too_strict_slo_has_no_eligible_concurrency(self):
        d=analyze(EVIDENCE,slo_thresholds=(10.,))
        self.assertIsNone(d["slo_decisions"][0]["selected_concurrency"])
        self.assertEqual(d["slo_decisions"][0]["eligible_concurrencies"],[])

    def test_missing_seed_rejected_not_estimated(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for item in EVIDENCE.glob("mixed_c*_seed*.json"):
                if item.name != "mixed_c8_seed137.json":
                    (base/item.name).write_bytes(item.read_bytes())
            with self.assertRaises(FileNotFoundError):
                analyze(base)

    def test_tampered_success_gate_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for item in EVIDENCE.glob("mixed_c*_seed*.json"):
                (base/item.name).write_bytes(item.read_bytes())
            p=base/"mixed_c4_seed42.json"
            d=json.loads(p.read_text())
            d["summary"]["succeeded"]=0
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(base)


if __name__=="__main__":
    unittest.main()
