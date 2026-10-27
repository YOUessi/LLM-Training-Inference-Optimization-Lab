"""Validate pinned real vLLM results and failed long-decode protocol."""
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "docs" / "evidence"


def load(path: str):
    with (ROOT / path).open(encoding="utf-8") as f:
        return json.load(f)


class Stage3RealVLLMResults(unittest.TestCase):
    def test_first_real_server_uses_vllm(self):
        s = load("stage3_vllm_summary.json")
        self.assertEqual(s["status"], "measured")
        self.assertEqual(s["engine_label"], "vllm")
        self.assertEqual(s["kind"], "openai_compatible_sse_concurrency_sweep")
        self.assertEqual([x["concurrency"] for x in s["cases"]], [1,2,4])
        for case in s["cases"]:
            self.assertEqual(case["summary"]["succeeded"], 8)
            self.assertEqual(case["summary"]["failed"], 0)
            self.assertEqual(case["summary"]["usage_coverage"], 1.0)

    def test_repeated_short_answers_have_real_21_token_counts(self):
        total = 0
        for group, order in (("short_forward",[1,2,4,8]),
                             ("short_reverse",[8,4,2,1])):
            s = load(f"stage3_vllm_repeated/{group}/summary.json")
            self.assertEqual(s["status"], "measured")
            self.assertEqual([r["concurrency"] for r in s["cases"]],order)
            for c in order:
                d = load(f"stage3_vllm_repeated/{group}/concurrency_{c}.json")
                self.assertEqual(d["summary"]["succeeded"], 24)
                self.assertEqual(d["summary"]["failed"], 0)
                self.assertEqual(d["summary"]["usage_coverage"],1.0)
                self.assertTrue(all(r["completion_tokens"]==21 for r in d["requests"]))
                total += d["summary"]["succeeded"]
        self.assertEqual(total,192)

    def test_true_long_output_128_and_length_gate(self):
        s=load("stage3_vllm_repeated/long_decode/summary.json")
        self.assertEqual(s["status"], "measured")
        self.assertEqual(s["engine_label"], "vllm")
        self.assertEqual(s["workload"]["max_output_tokens"],128)
        self.assertEqual(s["workload"]["minimum_actual_mean_completion_tokens"],90)
        total=0
        for case in s["cases"]:
            self.assertEqual(case["actual_mean_completion_tokens"],128.)
            self.assertGreaterEqual(case["actual_mean_completion_tokens"],90.)
            c=case["concurrency"]
            d=load(f"stage3_vllm_repeated/long_decode/concurrency_{c}.json")
            self.assertEqual(d["summary"]["usage_coverage"],1.)
            self.assertTrue(all(r["status"]=="ok" and
                                r["completion_tokens"]==128 for r in d["requests"]))
            total += len(d["requests"])
        self.assertEqual(total,36)

    def test_invalid_long_experiment_kept_as_negative(self):
        s=load("stage3_vllm_invalid_long/summary.json")
        self.assertEqual(s["workload"]["max_output_tokens"],128)
        # Legacy status was "measured" for real requests but does not prove long decode.
        for c in (1,4,8):
            d=load(f"stage3_vllm_invalid_long/concurrency_{c}.json")
            self.assertTrue(all(r["completion_tokens"]==21 for r in d["requests"]))
        valid=load("stage3_vllm_repeated/long_decode/summary.json")
        self.assertNotEqual(
            s["workload"].get("response_instruction_sha256"),
            valid["workload"]["response_instruction_sha256"]
        )

if __name__=="__main__":
    unittest.main()
