"""CPU-only checks for the DDP exact-recovery validation harness."""
from __future__ import annotations

import unittest

from llm_opt_lab.training.ddp_train import parse_args as ddp_args
from llm_opt_lab.training.ddp_train import model_sha256
from llm_opt_lab.provenance import environment
from scripts.verify_ddp_resume import parse_args


class RecoveryContractTests(unittest.TestCase):
    def test_driver_parser(self):
        args = parse_args(["--output-dir", "/tmp/new_experiment", "--device", "cpu",
                           "--steps", "8", "--split", "3"])
        self.assertEqual(args.split, 3)
        self.assertEqual(args.steps, 8)

    def test_invalid_step_partition(self):
        for split in ("0", "8", "9"):
            with self.subTest(split=split):
                with self.assertRaises(SystemExit):
                    parse_args(["--output-dir", "/tmp/new_experiment",
                                "--steps", "8", "--split", split])

    def test_ddp_parser_total_steps(self):
        args = ddp_args(["--steps", "8", "--device", "cuda"])
        self.assertEqual(args.steps, 8)
        self.assertEqual(args.device, "cuda")


if __name__ == "__main__":
    unittest.main()
