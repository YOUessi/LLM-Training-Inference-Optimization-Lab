"""Open-loop arrival scheduling, real-token gates and honest queue accounting."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage8_arrivals import (
    MODES, RATES, SEEDS, frozen_requests, group_language,
    output_length_checks, parse_args, read_kv_cache_utilization, schedule,
)


ROOT=Path(__file__).resolve().parents[1]/"data"


class Stage8ArrivalContracts(unittest.TestCase):
    def test_pin_inputs_and_corpus_counts(self):
        rows,meta=frozen_requests(
            ROOT/"stage8_frozen_requests.jsonl",
            ROOT/"stage8_frozen_requests_manifest.json",
        )
        self.assertEqual(len(rows),24)
        self.assertEqual(len({r["request_id"] for r in rows}),24)
        self.assertEqual(set(meta["generation_instructions"]),set(MODES))

    def test_poisson_arrival_replay_and_rate_validation(self):
        a=schedule(42,2.,24)
        b=schedule(42,2.,24)
        c=schedule(137,2.,24)
        self.assertEqual(a,b)
        self.assertNotEqual(a,c)
        self.assertEqual(a[0],0.)
        self.assertEqual(len(a),24)
        self.assertTrue(all(x<y for x,y in zip(a,a[1:])))
        with self.assertRaises(ValueError):
            schedule(42,0,24)

    def test_fixed_window_controls_realized_mean_rate_and_preserves_bursts(self):
        for seed in (42,137):
            for rate in (2.,8.):
                xs=schedule(seed,rate,24,kind="fixed_window")
                self.assertAlmostEqual(23/xs[-1],rate)
                self.assertTrue(all(a<b for a,b in zip(xs,xs[1:])))
        self.assertNotEqual(schedule(42,8.,24,kind="fixed_window"),
                            schedule(137,8.,24,kind="fixed_window"))
        with self.assertRaises(ValueError):
            schedule(42,8.,24,kind="unknown")

    def test_length_modes_are_real_token_gate(self):
        fake=[{"status":"ok","completion_tokens":12} for _ in range(24)]
        self.assertEqual(output_length_checks("label",fake)["status"],"pass")
        self.assertEqual(output_length_checks("detailed",fake)["status"],
                         "fail_realized_length_gate")
        long=[{"status":"ok","completion_tokens":128} for _ in range(24)]
        self.assertEqual(output_length_checks("summary",long)["status"],
                         "fail_realized_length_gate")
        self.assertEqual(output_length_checks("detailed",long)["status"],"pass")

    def test_kv_usage_metric_not_assumed_present(self):
        self.assertIsNone(read_kv_cache_utilization("# HELP other stuff\n"))
        self.assertAlmostEqual(
            read_kv_cache_utilization(
                'vllm:kv_cache_usage_perc{engine="0"} 0.23\n'),.23
        )
        self.assertIsNone(
            read_kv_cache_utilization('vllm:kv_cache_usage_perc 1.4\n')
        )

    def test_language_breakdown(self):
        rows=[]
        for lang in ("en","zh-cn","ja"):
            rows.append({
                "language":lang,"status":"ok",
                "ttft_ms":11.,"ttft_from_due_ms":20.,
                "prompt_tokens":300,
            })
        d=group_language(rows)
        self.assertEqual(set(d),{"en","zh-cn","ja"})
        self.assertTrue(all(q["succeeded"]==1 for q in d.values()))

    def test_closed_parameter_space(self):
        good=parse_args(["--model","qwen","--mode","summary",
                         "--rate","8","--seed","137",
                         "--output","/tmp/stage8.json"])
        self.assertEqual((good.mode,good.rate,good.seed),("summary",8.,137))
        for args in (["--rate","10"],["--seed","999"],
                     ["--mode","unknown"],["--warmup","3"]):
            with self.subTest(args=args):
                with self.assertRaises(SystemExit):
                    parse_args(["--model","qwen","--mode","label",
                                "--rate","2","--seed","42",
                                "--output","/tmp/stage8.json"]+args)


if __name__=="__main__":
    unittest.main()
