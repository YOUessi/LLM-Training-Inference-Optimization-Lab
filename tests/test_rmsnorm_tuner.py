"""Parse and candidate precondition tests; no GPU required."""
import unittest
from llm_opt_lab.kernels.rmsnorm_tune import parse_args


class TunerArgsTests(unittest.TestCase):
    def test_default(self):
        args = parse_args([])
        self.assertEqual((args.rows, args.cols, args.trials), (128, 4096, 4))
        self.assertEqual(args.warps, (1, 2, 4, 8))

    def test_sorted_deduplicated(self):
        args = parse_args(["--warps", "8,4,2,4", "--rows-per-program", "4,1,2"])
        self.assertEqual(args.warps, (2, 4, 8))
        self.assertEqual(args.rows_per_program, (1, 2, 4))

    def test_invalid(self):
        for opts in (["--trials", "1"], ["--rows", "0"], ["--warps", "3"],
                     ["--eps", "-0.01"], ["--cols", "20000"]):
            with self.subTest(opts=opts):
                with self.assertRaises(SystemExit):
                    parse_args(opts)
