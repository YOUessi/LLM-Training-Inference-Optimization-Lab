"""Prefix-cache counter acceptance gates, including contamination regressions."""
from __future__ import annotations

import unittest

from llm_opt_lab.serving.prefix_cache_metrics import analyze, parse_totals


def prom(queries: int, hits: int) -> str:
    return (
        "# HELP vllm:prefix_cache_queries_total Prefix cache queries\n"
        'vllm:prefix_cache_queries_total{engine="0",model_name="local"} '
        f"{queries}.0\n"
        "# HELP vllm:prefix_cache_hits_total Prefix cache hits\n"
        'vllm:prefix_cache_hits_total{engine="0",model_name="local"} '
        f"{hits}.0\n"
    )


class PrefixCacheMetricsTests(unittest.TestCase):
    def test_shared_warm_cache_has_high_hits(self):
        r = analyze(prom(1000, 500), prom(32000, 30000),
                    cache_state="on", mode="shared")
        self.assertEqual(r["status"], "pass")
        self.assertGreater(r["hit_fraction"], 0.9)

    def test_distinct_prefix_negative_control_passes_low_hit_rate(self):
        r = analyze(prom(1000, 100), prom(32000, 700),
                    cache_state="on", mode="distinct")
        self.assertEqual(r["status"], "pass")
        self.assertLess(r["hit_fraction"], 0.1)

    def test_repeated_distinct_ids_are_rejected(self):
        # Exactly the cross-arm contamination observed on Tang.
        r = analyze(prom(1000, 100), prom(32000, 31000),
                    cache_state="on", mode="distinct")
        self.assertEqual(r["status"], "fail_distinct_prefix_reused")

    def test_off_counters_are_zero(self):
        r = analyze(prom(0, 0), prom(0, 0),
                    cache_state="off", mode="shared")
        self.assertEqual(r["status"], "pass")
        self.assertIsNone(r["hit_fraction"])

    def test_off_nonzero_count_invalid(self):
        r = analyze(prom(0, 0), prom(1000, 10),
                    cache_state="off", mode="distinct")
        self.assertEqual(r["status"], "fail_caching_off_counters_nonzero")

    def test_missing_metrics_fail_closed(self):
        r = analyze("# Nothing\n", prom(500, 0),
                    cache_state="on", mode="shared")
        self.assertEqual(r["status"], "invalid_missing_counters")

    def test_counter_reset_rejected(self):
        r = analyze(prom(5000, 3000), prom(100, 20),
                    cache_state="on", mode="shared")
        self.assertEqual(r["status"], "invalid_counter_reset_or_mismatch")


if __name__ == "__main__":
    unittest.main()
