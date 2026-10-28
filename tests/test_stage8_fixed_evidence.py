"""Verify complete 288-request Stage8 matched-rate GPU evidence from raw JSON."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage8_analysis import analyze


ROOT = Path(__file__).resolve().parents[1] / "docs/evidence/stage8_fixed_window"
SOURCE = "2c6a7d008a89a24af26962b7fb6c38170e0714c7"


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class Stage8MatchedRateEvidenceTests(unittest.TestCase):
    def test_provenance_and_hashes(self):
        meta=load("fixed_manifest.json")
        self.assertEqual(meta["schema"],"stage8-fixed-window-arrivals-evidence-v1")
        self.assertEqual(meta["source_commit"],SOURCE)
        self.assertEqual(meta["real_requested_count"],288)
        self.assertFalse(meta["is_unconditioned_poisson"])
        self.assertTrue(meta["actual_mean_offered_rate_matched_both_seeds"])
        self.assertTrue(meta["cache_off_confirmed_in_actual_vllm_log"])
        self.assertTrue(meta["server_log_retained_only_on_Tang"])
        self.assertEqual(len(meta["raw_files_sha256"]),12)
        for filename,digest in meta["raw_files_sha256"].items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    hashlib.sha256((ROOT/"fixed"/filename).read_bytes()).hexdigest(),
                    digest
                )
        self.assertEqual(
            hashlib.sha256((ROOT/"fixed_analysis.json").read_bytes()).hexdigest(),
            meta["analysis_sha256"]
        )

    def test_actual_matched_rate_and_output_modes(self):
        total=0
        for seed in (42,137):
            for mode in ("label","summary","detailed"):
                for rate in (2,8):
                    name=f"fixed/{mode}_rate{rate}_seed{seed}.json"
                    p=load(name)
                    with self.subTest(name=name):
                        self.assertEqual(p["status"],"measured")
                        self.assertEqual(p["kind"],"real_vllm_stage8_multilingual_openloop")
                        self.assertEqual(p["environment"]["git_commit"],SOURCE)
                        self.assertEqual(p["config"]["schedule_kind"],"fixed_window")
                        self.assertEqual(p["summary"]["succeeded"],24)
                        self.assertEqual(p["summary"]["failed"],0)
                        self.assertEqual(p["summary"]["usage_coverage"],1.)
                        self.assertEqual(p["summary"]["prompt_usage_coverage"],1.)
                        self.assertEqual(p["length_gates"]["status"],"pass")
                        self.assertTrue(p["all_actual_input_tokens_in_bounds"])
                        self.assertEqual(len(p["requests"]),24)
                        self.assertAlmostEqual(
                            p["offered_arrival_rate_empirical_per_second"],rate,places=8
                        )
                        self.assertTrue(p["gpu_kv_cache_gauge"]["available"])
                        self.assertGreater(p["gpu_kv_cache_gauge"]["samples"],0)
                        for lang in ("en","zh-cn","ja"):
                            self.assertEqual(p["by_language"][lang]["sent"],8)
                            self.assertEqual(p["by_language"][lang]["succeeded"],8)
                        if mode=="label":
                            self.assertLess(p["length_gates"]["realized_mean"],10)
                        elif mode=="summary":
                            self.assertLess(p["length_gates"]["realized_mean"],100)
                        else:
                            self.assertEqual(p["length_gates"]["realized_mean"],128.)
                        total+=24
        self.assertEqual(total,288)

    def test_recompute_analysis_identically_from_raw(self):
        original=load("fixed_analysis.json")
        computed=analyze(ROOT/"fixed", expected_source_sha=SOURCE,
                         expected_schedule_kind="fixed_window")
        self.assertEqual(original,computed)
        self.assertEqual(original["measured_requests"],288)
        self.assertEqual(original["case_count"],12)
        self.assertEqual(len(original["descriptive_two_seed_cells"]),6)
        self.assertTrue(original["is_randomized_fixed_duration_design"])

    def test_first_unconditioned_different_experiment_retained(self):
        old_root=ROOT.parent/"stage8_unconditioned"
        old=json.loads((old_root/"unconditioned_analysis.json").read_text())
        self.assertEqual(old["measured_requests"],288)
        actual={round(x["empirical_offered_rate"],2) for x in old["per_case"]}
        self.assertEqual(actual,{1.48,2.86,5.93,11.45})


if __name__ == "__main__":
    unittest.main()
