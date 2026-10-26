"""FSDP smoke CLI checks, no torch/GPU imported during test collection."""
from __future__ import annotations

import unittest

from llm_opt_lab.training.fsdp_smoke import parse_args


class FSDPArgumentsTests(unittest.TestCase):
    def test_valid(self):
        args = parse_args(["--steps", "10", "--features", "128", "--hidden", "256"])
        self.assertEqual(args.steps, 10)
        self.assertEqual(args.hidden, 256)

    def test_bad_dimensions(self):
        for args in (
            ["--steps", "0"],
            ["--batch-size", "-1"],
            ["--features", "0"],
            ["--lr", "0"],
        ):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                parse_args(args)


if __name__ == "__main__":
    unittest.main()
