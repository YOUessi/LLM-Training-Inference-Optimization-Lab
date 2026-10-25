"""Offline Transformers benchmark contracts; requires no model or GPU."""
from __future__ import annotations

import unittest

from llm_opt_lab.serving.hf_baseline import (
    actual_generated_tokens, parse_args, quantile
)


class _FakeTensor:
    def __init__(self, values):
        self.values=values

    def tolist(self):
        return self.values


class OfflineGenerationArgsTests(unittest.TestCase):
    def test_default_batch_sweep(self):
        a=parse_args([])
        self.assertEqual(a.batch_sizes,(1,2,4))
        self.assertEqual(a.repeats,3)

    def test_reject_invalid(self):
        for arg in [
            ["--batch-sizes","2,2"],
            ["--batch-sizes","0"],
            ["--batch-sizes","a"],
            ["--repeats","1"],
            ["--max-new-tokens","0"],
            ["--max-new-tokens","300"],
        ]:
            with self.subTest(arg=arg):
                with self.assertRaises(SystemExit):
                    parse_args(arg)

    def test_eos_token_not_overcounted(self):
        self.assertEqual(actual_generated_tokens(_FakeTensor([5,7,2,2,2]),{2}),3)
        self.assertEqual(actual_generated_tokens(_FakeTensor([1,4,9]),{2}),3)

    def test_latency_quantile(self):
        self.assertAlmostEqual(quantile([10.,20.],0.95),19.5)


if __name__ == "__main__":
    unittest.main()
