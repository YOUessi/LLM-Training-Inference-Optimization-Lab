"""Validate Stage-2 committed raw evidence without importing CUDA dependencies."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence"


class Stage2EvidenceTests(unittest.TestCase):
    def read(self, name: str):
        with (EVIDENCE / name).open(encoding="utf-8") as f:
            return json.load(f)

    def test_cpu_and_cuda_exact_resume(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                data = self.read(f"stage2_ddp_{device}_resume.json")
                self.assertEqual(data["status"], "pass")
                self.assertEqual(data["configuration"]["world_size"], 1)
                self.assertEqual(data["configuration"]["steps"], 8)
                self.assertEqual(data["configuration"]["split"], 3)
                self.assertTrue(all(data["equality"].values()))
                self.assertEqual(data["model_sha256_full"], data["model_sha256_resumed"])

    def test_real_streaming_is_not_mislabeled_vllm(self):
        data = self.read("stage2_stream_summary.json")
        self.assertEqual(data["status"], "measured")
        self.assertEqual(data["kind"], "openai_compatible_sse_concurrency_sweep")
        self.assertEqual(data["engine_label"], "transformers_serialized")
        self.assertEqual([c["concurrency"] for c in data["cases"]], [1, 2, 4])
        for case in data["cases"]:
            with self.subTest(concurrency=case["concurrency"]):
                s = case["summary"]
                self.assertEqual(s["sent"], 8)
                self.assertEqual(s["succeeded"], 8)
                self.assertEqual(s["failed"], 0)
                self.assertEqual(s["usage_coverage"], 1.0)
                self.assertGreater(s["p95_ttft_ms"], 0)
                self.assertGreater(s["output_tokens_per_second"], 0)
                raw = self.read(f"stage2_stream_c{case['concurrency']}.json")
                self.assertEqual(raw["kind"], "serving_sse_benchmark")
                self.assertEqual(raw["summary"]["succeeded"], 8)

    def test_provenance_explicitly_distinguishes_snapshot_from_git_head(self):
        data = self.read("stage2_source_manifest.json")
        self.assertEqual(data["schema"], "stage2-experiment-provenance-v1")
        self.assertEqual(len(data["produced_files"]), 6)
        self.assertIn("Transformers", "Transformers")
        self.assertNotEqual(
            data["experiment_git_head_at_execution"],
            data["source_provenance"]["transformers_stream_server_commit"],
        )


if __name__ == "__main__":
    unittest.main()
