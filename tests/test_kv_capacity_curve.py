"""Capacity matrix accepts observed retention as data, not a failed eviction."""
from __future__ import annotations

import unittest

from llm_opt_lab.serving.kv_capacity_curve import classify, parse_args


class CapacityCurveTests(unittest.TestCase):
    def test_evicted_after_pressure_valid(self):
        p=classify({"cold":0.,"warm":.99,"after_pressure":.01,"after_rewarm":.99})
        self.assertEqual(p["post_pressure_classification"],"evicted")
        self.assertTrue(p["stage_valid"])

    def test_retained_after_pressure_valid(self):
        p=classify({"cold":0.,"warm":.99,"after_pressure":.95,"after_rewarm":.99})
        self.assertEqual(p["post_pressure_classification"],"retained")
        self.assertTrue(p["stage_valid"])

    def test_partial_after_pressure_valid(self):
        p=classify({"cold":0.,"warm":.99,"after_pressure":.50,"after_rewarm":.99})
        self.assertEqual(p["post_pressure_classification"],"partial")
        self.assertTrue(p["stage_valid"])

    def test_invalid_warmup_or_missing_stage(self):
        self.assertFalse(classify({
            "cold":.90,"warm":.99,"after_pressure":.01,"after_rewarm":.99
        })["stage_valid"])
        with self.assertRaises(ValueError):
            classify({"cold":0,"warm":.99})

    def test_blocks_and_fillers_validate(self):
        a=parse_args(["--model","qwen","--blocks","512","--fillers","4","--output","/tmp/t.json"])
        self.assertEqual((a.blocks,a.fillers),(512,4))
        with self.assertRaises(SystemExit):
            parse_args(["--model","qwen","--blocks","8","--fillers","4","--output","/tmp/t.json"])


if __name__=="__main__":
    unittest.main()
