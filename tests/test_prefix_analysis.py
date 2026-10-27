"""Offline statistical summaries and fail-closed pair matching for Prefix Cache."""
from __future__ import annotations

import unittest

from llm_opt_lab.serving.prefix_analysis import summarize_pair, aggregate_two_seeds


def row(seed=42, mode="shared", concurrency=1, ttft=50.,
        tps=100., prompt=1209., hits=.95):
    return {
        "seed":seed,"mode":mode,"concurrency":concurrency,
        "p95_ttft_ms":ttft,"output_tokens_per_second":tps,
        "mean_prompt_tokens":prompt, "metric":{"hit_fraction":hits},
    }


class PrefixCacheAnalysisTests(unittest.TestCase):
    def test_pair_speedup_correct_direction(self):
        p=summarize_pair(
            row(ttft=80.,tps=100.,hits=None),
            row(ttft=20.,tps=300.,hits=.95)
        )
        self.assertAlmostEqual(p["ttft_off_over_on"],4.)
        self.assertAlmostEqual(p["throughput_on_over_off"],3.)

    def test_prompt_length_mismatch_blocks_pair(self):
        with self.assertRaises(ValueError):
            summarize_pair(row(prompt=1200.),row(prompt=1220.))

    def test_unmatched_pair_blocks(self):
        with self.assertRaises(ValueError):
            summarize_pair(row(seed=42),row(seed=137))

    def test_two_seed_descriptive_aggregates(self):
        pairs=[]
        for mode in ("shared","distinct"):
            for c in (1,8):
                for seed in (42,137):
                    a=row(seed,mode,c,ttft=80.,tps=100.,hits=None)
                    b=row(seed,mode,c,
                          ttft=20. if mode=="shared" else 80.,
                          tps=300. if mode=="shared" else 100.,
                          hits=.95 if mode=="shared" else .02)
                    pairs.append(summarize_pair(a,b))
        totals=aggregate_two_seeds(pairs)
        self.assertEqual(len(totals),4)
        self.assertAlmostEqual(totals[0]["mean_ttft_off_over_on"],4.)
        self.assertAlmostEqual(totals[2]["mean_ttft_off_over_on"],1.)
        self.assertAlmostEqual(totals[2]["mean_prefix_cache_hit_fraction_on"],.02)


if __name__=="__main__":
    unittest.main()
