"""Validate published Stage 4 real Prefix Cache negative-controls and sources."""
from __future__ import annotations

import hashlib
import json
import statistics
import unittest
from pathlib import Path

from llm_opt_lab.serving.prefix_analysis import summarize_pair


ROOT = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "stage4_prefix_cache"


def read(name: str):
    with (ROOT / name).open(encoding="utf-8") as f:
        return json.load(f)


class Stage4EvidenceTests(unittest.TestCase):
    def test_all_real_cases_pinned_and_sha256_valid(self):
        summary = read("stage4_prefix_cache_analysis.json")
        manifest = read("stage4_prefix_cache_server_manifest.json")
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["measured_real_requests"], 384)
        self.assertEqual(len(summary["arms"]), 16)
        self.assertEqual(len(summary["matched_on_off_pairs"]), 8)
        self.assertEqual(
            summary["source_git_commit"], manifest["experiment_source_commit"]
        )
        self.assertEqual(len(manifest["corrected_case_file_sha256"]), 32)
        for rel, sha in manifest["corrected_case_file_sha256"].items():
            with self.subTest(file=rel):
                data=(ROOT/"stage4_prefix_cache_raw"/rel).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), sha)

    def test_negative_control_and_shared_hit_acceptance(self):
        for state in ("off", "on"):
            for mode in ("shared","distinct"):
                for seed in (42,137):
                    for c in (1,8):
                        label=f"{mode}-c{c}-seed{seed}"
                        m=read(f"stage4_prefix_cache_raw/{state}/{label}-metrics.json")
                        r=read(f"stage4_prefix_cache_raw/{state}/{label}.json")
                        with self.subTest(state=state,mode=mode,seed=seed,c=c):
                            self.assertEqual(m["status"],"pass")
                            self.assertEqual(r["status"],"measured")
                            self.assertEqual(r["summary"]["succeeded"],24)
                            self.assertEqual(r["summary"]["failed"],0)
                            self.assertEqual(r["summary"]["prompt_usage_coverage"],1.0)
                            self.assertEqual(r["summary"]["usage_coverage"],1.0)
                            self.assertTrue(r["prompt_length_gate"]["all_lengths_within_bounds"])
                            self.assertTrue(r["config"]["distinct_prefix_ids_isolated_by_concurrency"])
                            self.assertEqual(r["environment"]["git_commit"],
                                             "d3f34614f133f9d9b5f071944abb472510fc99f9")
                            if state=="off":
                                self.assertEqual(m["delta_queried_tokens"],0)
                                self.assertEqual(m["delta_hit_tokens"],0)
                            elif mode=="shared":
                                self.assertGreaterEqual(m["hit_fraction"],.70)
                            else:
                                self.assertLessEqual(m["hit_fraction"],.10)

    def test_two_seed_matched_cache_effect(self):
        d=read("stage4_prefix_cache_analysis.json")
        pairs=d["matched_on_off_pairs"]
        for p in pairs:
            self.assertTrue(900 <= p["mean_prompt_tokens"] <= 1450)
            self.assertGreater(p["ttft_off_over_on"],0)
        shared=[x for x in pairs if x["mode"]=="shared"]
        distinct=[x for x in pairs if x["mode"]=="distinct"]
        self.assertEqual(len(shared),4)
        self.assertEqual(len(distinct),4)
        self.assertTrue(all(x["ttft_off_over_on"] > 3 for x in shared))
        self.assertTrue(all(0.85 <= x["ttft_off_over_on"] <= 1.1 for x in distinct))
        self.assertGreater(statistics.mean(x["on_metric"]["hit_fraction"] for x in shared),.90)
        self.assertLess(statistics.mean(x["on_metric"]["hit_fraction"] for x in distinct),.05)

    def test_first_negative_control_invalid_run_is_preserved(self):
        d=read("stage4_invalid_first_run_summary.json")
        self.assertEqual(d["status"],"invalid_negative_control_contamination")
        self.assertEqual(len(d["case_rows"]),16)
        bad=[x for x in d["case_rows"] if x["metrics"]["status"]=="fail_distinct_prefix_reused"]
        self.assertEqual(len(bad),2)
        self.assertEqual(
            {(x["seed"],x["mode"],x["concurrency"]) for x in bad},
            {(42,"distinct",8),(137,"distinct",1)},
        )


if __name__=="__main__":
    unittest.main()
