"""Argument smoke tests need no torch or CUDA."""
from __future__ import annotations
import unittest
from llm_opt_lab.kernels.benchmark import parse_args as kernel_args
from llm_opt_lab.serving.bench_client import parse_args as serving_args
from llm_opt_lab.training.ddp_train import parse_args as ddp_args


class CLIArgumentTests(unittest.TestCase):
    def test_gpu_arguments_no_gpu_required(self):
        a=kernel_args(["--rows","64","--cols","768"])
        self.assertEqual((a.rows,a.cols),(64,768))

    def test_gpu_args_reject_invalid_shape(self):
        with self.assertRaises(SystemExit):
            kernel_args(["--rows","0"])

    def test_serving_rejects_negative_concurrency(self):
        with self.assertRaises(SystemExit):
            serving_args(["--model","fake","--concurrency","-4"])

    def test_ddp_default_is_cpu(self):
        self.assertEqual(ddp_args(["--steps","2"]).device,"cpu")


if __name__=="__main__":
    unittest.main()
