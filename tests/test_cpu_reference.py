"""Tests run on Mac and GitHub Actions, no third-party dependencies."""
from __future__ import annotations

import math
import unittest

from llm_opt_lab.kernels.cpu_reference import rmsnorm_row, softmax_row


class CPUReferenceTests(unittest.TestCase):
    def test_rmsnorm_unit_scale(self):
        xs = [3., 4.]
        expected = [x / math.sqrt(12.5 + 1e-6) for x in xs]
        for value, reference in zip(rmsnorm_row(xs), expected):
            self.assertAlmostEqual(value, reference, places=12)

    def test_rmsnorm_zeros(self):
        self.assertEqual(rmsnorm_row([0.0, 0.0]), [0.0, 0.0])

    def test_rmsnorm_rejects_empty(self):
        with self.assertRaises(ValueError):
            rmsnorm_row([])

    def test_rmsnorm_rejects_bad_eps(self):
        for e in [0., -1., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                rmsnorm_row([1.0], e)

    def test_softmax_stable_huge_values(self):
        got = softmax_row([10_000., 10_001.])
        self.assertAlmostEqual(sum(got), 1.0, places=12)
        self.assertGreater(got[1], got[0])

    def test_softmax_translation_invariance(self):
        got = softmax_row([1., 4., -2.])
        shifted = softmax_row([1001., 1004., 998.])
        for a,b in zip(got, shifted):
            self.assertAlmostEqual(a, b, places=12)

    def test_softmax_rejects_nonfinite(self):
        for x in [float("nan"), float("inf"), float("-inf")]:
            with self.assertRaises(ValueError):
                softmax_row([x, 0.])

    def test_softmax_rejects_empty(self):
        with self.assertRaises(ValueError):
            softmax_row([])


if __name__ == "__main__":
    unittest.main()
