"""Offline unit checks for cache-on/off prefill workload controls."""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_opt_lab.serving.prefix_cache_bench import (
    make_prompt, parse_args, verify_request_lengths,
)


class PrefixCacheWorkloadTests(unittest.TestCase):
    def test_shared_prefix_is_constant_across_request_indices(self):
        a = make_prompt(0, seed=42, repetitions=35, mode="shared")
        b = make_prompt(8, seed=42, repetitions=35, mode="shared")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("V00000000"))
        self.assertGreater(len(a), 4000)

    def test_distinct_prefix_changes_at_first_content(self):
        a = make_prompt(0, seed=42, repetitions=35, mode="distinct")
        b = make_prompt(1, seed=42, repetitions=35, mode="distinct")
        self.assertNotEqual(a[:10], b[:10])
        self.assertEqual(a[10:], b[10:])
        c = make_prompt(0, seed=137, repetitions=35, mode="distinct")
        self.assertNotEqual(a[:10], c[:10])

    def test_min_max_server_reported_token_gate(self):
        records = [
            {"status":"ok", "prompt_tokens":1201},
            {"status":"ok", "prompt_tokens":1205},
        ]
        good, detail = verify_request_lengths(records, lower=900, upper=1450)
        self.assertTrue(good)
        self.assertEqual(detail["mean_actual_prompt_tokens"], 1203.)
        bad, detail = verify_request_lengths(records, lower=1250, upper=1450)
        self.assertFalse(bad)
        self.assertFalse(detail["all_lengths_within_bounds"])

    def test_missing_prompt_usage_must_not_succeed(self):
        good, detail = verify_request_lengths(
            [{"status":"ok", "prompt_tokens":None}], lower=900, upper=1450
        )
        self.assertFalse(good)
        self.assertFalse(detail["prompt_usage_complete"])

    def test_argument_rejections(self):
        for opt in (
            ["--prefix-repetitions", "70"],
            ["--concurrency", "0"],
            ["--min-prompt-tokens", "1600", "--max-prompt-tokens", "1450"],
            ["--max-new-tokens", "512"],
        ):
            with self.subTest(opt=opt):
                with self.assertRaises(SystemExit):
                    parse_args(["--model","x", "--mode","shared",
                                "--declared-cache-state","off",
                                "--output","/tmp/t.json"]+opt)

    def test_correct_control_surface(self):
        a = parse_args(["--model","qwen",
                        "--mode","distinct","--declared-cache-state","on",
                        "--output","/tmp/test.json"])
        self.assertEqual(a.prefix_repetitions,35)
        self.assertEqual(a.requests,24)
        self.assertEqual(a.min_prompt_tokens,900)


if __name__ == "__main__":
    unittest.main()
