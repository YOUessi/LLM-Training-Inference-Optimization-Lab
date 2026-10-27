"""Stage 5 KV cache eviction and heterogeneous prompt benchmark CPU contracts."""
from __future__ import annotations

import unittest

from llm_opt_lab.serving.kv_capacity_probe import (
    parse_args as parse_capacity,
    ratio_between, evaluate,
)
from llm_opt_lab.serving.mixed_context_bench import (
    parse_args as parse_mixed,
    bucketize,
)
from llm_opt_lab.serving.prefix_cache_metrics import METRICS


def counters(q,h):
    return {METRICS[0]:float(q), METRICS[1]:float(h)}


class CacheEvictionContracts(unittest.TestCase):
    def test_valid_cache_counters(self):
        self.assertAlmostEqual(
            ratio_between(counters(100,10), counters(1100,910))["cache_hit_fraction"],
            0.9,
        )

    def test_counter_reset_and_missing_blocked(self):
        for before,after in [
            (counters(100,50),counters(90,40)),
            (counters(100,50),counters(200,300)),
            ({METRICS[0]:None,METRICS[1]:0},counters(300,20)),
        ]:
            with self.subTest(before=before):
                with self.assertRaises(ValueError):
                    ratio_between(before,after)

    def test_eviction_acceptance(self):
        events=[
            {"stage":"initial_miss","cache":{"cache_hit_fraction":.01}},
            {"stage":"immediate_replay","cache":{"cache_hit_fraction":.98}},
            {"stage":"post_pressure_replay","cache":{"cache_hit_fraction":.04}},
            {"stage":"restored_replay","cache":{"cache_hit_fraction":.97}},
        ]
        self.assertTrue(all(evaluate(events).values()))
        events[2]["cache"]["cache_hit_fraction"]=.80
        self.assertFalse(evaluate(events)["replay_after_pressure_below_25pct"])

    def test_eviction_argument_bounds(self):
        a=parse_capacity(["--model","qwen","--num-gpu-blocks-override","256",
                          "--output","/tmp/eviction.json"])
        self.assertEqual(a.distinct_fillers,12)
        for extra in (["--distinct-fillers","2"],["--prefix-repetitions","0"],
                      ["--num-gpu-blocks-override","4"]):
            with self.subTest(extra=extra),self.assertRaises(SystemExit):
                parse_capacity(["--model","qwen","--num-gpu-blocks-override","256",
                                "--output","/tmp/eviction.json"]+extra)


class MixedContextContracts(unittest.TestCase):
    def test_prompt_length_buckets(self):
        a=parse_mixed(["--model","qwen","--output","/tmp/mix.json"])
        self.assertEqual(a.repeats_per_length,6)
        self.assertEqual(a.concurrency,4)

    def test_buckets_reject_missing_or_wrong_input_tokens(self):
        specs={"short":(3,85,250),"long":(35,1050,1400)}
        records=[
            {"length_bucket":"short","status":"ok","prompt_tokens":150,
             "ttft_ms":12.,"latency_ms":40.},
            {"length_bucket":"long","status":"ok","prompt_tokens":1200,
             "ttft_ms":75.,"latency_ms":100.},
        ]
        d=bucketize(records,configurations=specs)
        self.assertEqual(d["short"]["status"],"pass")
        self.assertEqual(d["long"]["status"],"pass")
        records[1]["prompt_tokens"]=None
        self.assertEqual(bucketize(records,configurations=specs)["long"]["status"],"invalid")

    def test_out_of_bounds(self):
        with self.assertRaises(SystemExit):
            parse_mixed(["--model","qwen","--concurrency","16",
                         "--output","/tmp/mix.json"])


if __name__=="__main__":
    unittest.main()
