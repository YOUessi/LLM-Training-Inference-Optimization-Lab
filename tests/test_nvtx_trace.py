"""Argument validation for NVTX GPU trace driver, no GPU or PyTorch needed."""
from __future__ import annotations

import unittest

from llm_opt_lab.kernels.nvtx_trace import parse_args


class TraceArgsTests(unittest.TestCase):
    def test_default(self):
        a = parse_args([])
        self.assertEqual((a.rows, a.cols, a.dtype, a.repeats),
                         (256, 4096, "float32", 30))

    def test_valid_fixed_tensor(self):
        a = parse_args(["--rows","128","--cols","8192",
                        "--dtype","float16","--warmup","2","--repeats","8"])
        self.assertEqual(a.cols, 8192)
        self.assertEqual(a.repeats, 8)

    def test_rejects_invalid(self):
        for case in (
            ["--rows","0"],["--cols","0"],["--cols","20000"],
            ["--warmup","0"],["--repeats","0"],["--eps","nan"],
        ):
            with self.subTest(case=case):
                with self.assertRaises(SystemExit):
                    parse_args(case)


if __name__ == "__main__":
    unittest.main()
