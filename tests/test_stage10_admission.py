"""CPU-only correctness contract for all-offered admission/SLO accounting."""
from __future__ import annotations

import unittest

from llm_opt_lab.serving.stage10_admission import (
    OFFERED_COUNT, accounting, offered_slo_compliance, parse_args,
)


def record(i, ok=True, reason="queue_full"):
    common={
        "request_id":f"request-{i:02d}",
        "index":i,
        "language":("en","zh-cn","ja")[i%3],
        "client_executor_wait_ms":.2,
        "admission_queue_wait_ms": 2. if ok else 250.,
    }
    if ok:
        return {
            **common,"status":"ok",
            "ttft_from_due_ms":80.+i,
            "prompt_tokens":300,"completion_tokens":120,
        }
    return {
        **common,"status":"rejected","rejection_reason":reason,
        "ttft_from_due_ms":None,
        "prompt_tokens":None,"completion_tokens":None,
    }


class Stage10AdmissionAccountingTests(unittest.TestCase):
    def test_every_offered_request_in_slo_denominator(self):
        rows=[record(i,ok=i<18) for i in range(OFFERED_COUNT)]
        d=accounting(rows,4.)
        self.assertEqual(d["offered"],24)
        self.assertEqual(d["accepted"],18)
        self.assertEqual(d["completed"],18)
        self.assertEqual(d["rejected"],6)
        self.assertEqual(d["errors"],0)
        self.assertEqual(d["rejection_reasons"],{"queue_full":6})
        self.assertEqual(d["slo_all_offered"]["250"]["offered"],24)
        self.assertEqual(d["slo_all_offered"]["250"]["compliant_completed"],18)
        self.assertEqual(d["slo_all_offered"]["250"]["not_compliant"],6)
        self.assertEqual(d["slo_all_offered"]["250"]["compliance_fraction"],.75)
        self.assertIsNone(d["p95_all_offered_ttft_ms"])
        self.assertGreater(d["p95_completed_only_arrival_ttft_ms"],0)
        self.assertEqual(d["useful_output_tokens"],18*120)

    def test_no_rejection_full_observation(self):
        d=accounting([record(i) for i in range(24)],4.)
        self.assertEqual(d["completed"],24)
        self.assertEqual(d["rejected"],0)
        self.assertEqual(d["slo_all_offered"]["250"]["compliant_completed"],24)
        self.assertAlmostEqual(
            d["p95_all_offered_ttft_ms"],
            d["p95_completed_only_arrival_ttft_ms"]
        )

    def test_rejections_do_not_manufacture_100_percent_slo(self):
        d=accounting([record(i,ok=i<8,reason="deadline_expired") for i in range(24)],4.)
        self.assertEqual(d["slo_all_offered"]["1000"]["compliance_fraction"],8/24)
        self.assertEqual(d["rejection_reasons"],{"deadline_expired":16})

    def test_missing_or_duplicate_offer_is_invalid(self):
        with self.assertRaises(ValueError):
            accounting([record(i) for i in range(23)],4.)
        records=[record(i) for i in range(24)]
        records[-1]["request_id"]=records[0]["request_id"]
        with self.assertRaises(ValueError):
            accounting(records,4.)

    def test_unknown_rejection_reason_invalid(self):
        with self.assertRaises(ValueError):
            accounting([record(i,ok=i<23,reason="silent_drop") for i in range(24)],4.)

    def test_heldout_seeds_and_deadline_are_explicitly_allowed(self):
        from llm_opt_lab.serving.stage10_admission import parse_args
        for seed in (2027,3031):
            a=parse_args(["--model","qwen","--mode","detailed",
                          "--policy","bounded","--seed",str(seed),
                          "--offered-rate","8","--deadline-ms","150",
                          "--output","/tmp/stage11.json"])
            self.assertEqual((a.seed,a.offered_rate,a.deadline_ms),(seed,8.,150.))
        with self.assertRaises(SystemExit):
            parse_args(["--model","qwen","--mode","detailed",
                        "--policy","bounded","--seed","2027",
                        "--deadline-ms","999","--output","/tmp/t.json"])

    def test_cli_requires_controlled_mode_and_seed(self):
        valid=parse_args([
            "--model","qwen","--mode","detailed","--policy","bounded",
            "--seed","42","--output","/tmp/stage10.json",
        ])
        self.assertEqual((valid.mode,valid.policy,valid.seed),("detailed","bounded",42))
        with self.assertRaises(SystemExit):
            parse_args(["--model","qwen","--mode","unknown","--policy","bounded",
                        "--seed","42","--output","/tmp/stage10.json"])


if __name__=="__main__":
    unittest.main()
