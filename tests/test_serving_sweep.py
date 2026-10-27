"""Serving workload sweep contracts — CPU only."""
import unittest
from llm_opt_lab.serving.sweep import parse_args, mean_completion_tokens


class SweepParseTests(unittest.TestCase):
    def test_levels(self):
        args=parse_args(["--model", "qwen", "--concurrency", "4,1,8"])
        self.assertEqual(args.concurrency,(4,1,8))
        self.assertEqual(args.warmup_requests,2)

    def test_bad_levels(self):
        for text in ("", "0,1", "1,1", "abc", "1,65"):
            with self.subTest(text=text):
                with self.assertRaises(SystemExit):
                    parse_args(["--model","qwen","--concurrency",text])

    def test_actual_token_count_mean(self):
        rows = [
            {"status": "ok", "completion_tokens": 21},
            {"status": "ok", "completion_tokens": 25},
            {"status": "error", "completion_tokens": None},
        ]
        self.assertEqual(mean_completion_tokens(rows), 23)
        self.assertIsNone(mean_completion_tokens([{"status":"error", "completion_tokens":None}]))

    def test_minimum_actual_tokens_must_fit_budget(self):
        with self.assertRaises(SystemExit):
            parse_args(["--model", "qwen", "--max-tokens", "32",
                        "--min-average-completion-tokens", "90"])

    def test_bad_requests(self):
        with self.assertRaises(SystemExit):
            parse_args(["--model","qwen","--requests","0"])
