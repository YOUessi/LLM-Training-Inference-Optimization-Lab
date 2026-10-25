"""Serving workload sweep contracts — CPU only."""
import unittest
from llm_opt_lab.serving.sweep import parse_args


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

    def test_bad_requests(self):
        with self.assertRaises(SystemExit):
            parse_args(["--model","qwen","--requests","0"])
