"""CPU-only Stage9 analysis contracts; fixture data is NOT new GPU evidence."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage9_worker_analysis import SOURCE_SHA, analyze

EVIDENCE=Path(__file__).resolve().parents[1]/"docs/evidence/stage8_fixed_window/fixed"


def make_fake_cases(root: Path) -> None:
    for seed in (42,137):
        for mode,rate in (("label",8),("detailed",2),("detailed",8)):
            original=json.loads(
                (EVIDENCE/f"{mode}_rate{rate}_seed{seed}.json").read_text()
            )
            for workers in (8,32):
                d=json.loads(json.dumps(original))
                d["environment"]["git_commit"]=SOURCE_SHA
                d["config"]["max_client_workers"]=workers
                d["server_request_queue_gauges"]={
                    "available":True,"samples":5,
                    "max_server_waiting_requests": 0 if workers==8 else 3,
                    "max_server_running_requests":8,
                    "mean_server_waiting_requests": 0. if workers==8 else .8,
                }
                (root/f"{mode}_w{workers}_rate{rate}_seed{seed}.json").write_text(
                    json.dumps(d),encoding="utf-8"
                )


class Stage9WorkerAnalysisContracts(unittest.TestCase):
    def test_exact_match_and_no_invented_slo_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_fake_cases(root)
            a=analyze(root)
            self.assertEqual(a["status"],"pass")
            self.assertEqual(a["real_requests"],288)
            self.assertEqual(a["case_count"],12)
            self.assertEqual(len(a["matched_pairs"]),6)
            self.assertTrue(a["not_a_new_gpu_measurement"])
            self.assertTrue(all(p["p95_arrival_ttft_reduction_ms"]==0.
                                for p in a["matched_pairs"]))

    def test_bad_matching_schedule_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_fake_cases(root)
            f=root/"detailed_w32_rate8_seed42.json"
            d=json.loads(f.read_text())
            d["config"]["request_arrival_schedule_s"][0]=.37
            f.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(root)

    def test_missing_server_queue_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_fake_cases(root)
            f=root/"label_w8_rate8_seed137.json"
            d=json.loads(f.read_text())
            d["server_request_queue_gauges"]["available"]=False
            f.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(root)

    def test_missing_real_case_is_not_interpolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);make_fake_cases(root)
            (root/"detailed_w8_rate2_seed137.json").unlink()
            with self.assertRaises(FileNotFoundError):
                analyze(root)


if __name__=="__main__":
    unittest.main()
