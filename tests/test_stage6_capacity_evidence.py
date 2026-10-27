"""Pinned four real KV retention experiments must be numerically truthful."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]/"docs"/"evidence"/"stage6_capacity_matrix"


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding="utf-8"))


class Stage6RealGPUCapacityEvidence(unittest.TestCase):
    def test_all_classification_points(self):
        d=read("stage6_capacity_summary.json")
        self.assertEqual(d["status"],"pass")
        self.assertEqual(len(d["cases"]),4)
        mapping={(x["blocks"],x["fillers"]):x for x in d["cases"]}
        self.assertEqual({k:v["result"] for k,v in mapping.items()},{
            (128,4):"evicted",
            (256,4):"evicted",
            (512,4):"retained",
            (512,12):"evicted",
        })
        for k,x in mapping.items():
            self.assertGreater(x["post_pressure_ttft_ms"],0)
            if x["result"]=="retained":
                self.assertGreaterEqual(x["hit_fraction"],.75)
            else:
                self.assertLessEqual(x["hit_fraction"],.25)

    def test_raw_real_sse_and_cache_gate_consistency(self):
        for blocks,fillers in ((128,4),(256,4),(512,4),(512,12)):
            path=f"blocks{blocks}-fillers{fillers}/result.json"
            p=read(path)
            with self.subTest(path=path):
                self.assertEqual(p["status"],"measured")
                self.assertEqual(p["kind"],"stage6_kv_retention_curve_single_case")
                self.assertEqual(p["config"]["blocks"],blocks)
                self.assertEqual(p["config"]["fillers"],fillers)
                self.assertEqual(len(p["events"]),fillers+4)
                self.assertTrue(p["classification"]["stage_valid"])
                self.assertTrue(p["classification"]["initial_miss_valid"])
                self.assertTrue(p["classification"]["immediate_hit_valid"])
                self.assertTrue(p["classification"]["replay_hit_valid"])
                self.assertTrue(all(250<=x["prompt_tokens"]<=1700 for x in p["events"]))

    def test_manifest_source_commit_and_file_hashes(self):
        m=read("stage6_capacity_manifest.json")
        self.assertEqual(m["schema"],"stage6-capacity-matrix-provenance-v1")
        self.assertEqual(m["experiment_source_commit"],
                         "92f4e72a433ffb14262e0d9d9e449f37feec445d")
        self.assertEqual(len(m["source_files"]),4)
        self.assertTrue(m["raw_server_logs_stay_on_tang"])
        for f in m["source_files"]:
            self.assertTrue(f["effective_runtime_flags_verified"])
            self.assertEqual(len(f["server_log_sha256"]),64)
            self.assertEqual(
                hashlib.sha256((ROOT/f["path"]).read_bytes()).hexdigest(),
                f["sha256"],
            )


if __name__=="__main__":
    unittest.main()
