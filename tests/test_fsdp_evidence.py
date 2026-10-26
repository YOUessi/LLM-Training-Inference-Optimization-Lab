"""Check that committed FSDP evidence is correctly limited to single GPU."""
import json
import unittest
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence"


class FSDPEvidenceTests(unittest.TestCase):
    def read(self, filename):
        with (EVIDENCE / filename).open(encoding="utf-8") as f:
            return json.load(f)

    def test_single_gpu_fsdp_cannot_be_marked_as_sharded(self):
        p = self.read("stage2_fsdp_single_gpu.json")
        self.assertEqual(p["status"], "measured")
        self.assertEqual(p["kind"], "synthetic_fsdp1_training_smoke")
        self.assertTrue(p["not_multigpu_scaling_evidence"])
        self.assertTrue(p["not_an_llm_training_result"])
        self.assertEqual(p["config"]["world_size"], 1)
        self.assertEqual(p["config"]["steps"], 20)
        self.assertIn("NO_SHARD", p["config"]["effective_sharding_strategy"])
        self.assertGreater(p["results"]["median_step_ms"], 0)
        self.assertGreater(p["results"]["peak_allocated_gpu_gib"], 0)

    def test_same_shape_ddp_reference_is_independent(self):
        fsdp = self.read("stage2_fsdp_single_gpu.json")
        ddp = self.read("stage2_ddp_single_gpu_reference.json")
        self.assertEqual(ddp["status"], "measured")
        self.assertEqual(ddp["config"]["world_size"], 1)
        self.assertEqual(fsdp["config"]["features"], ddp["config"]["features"])
        self.assertEqual(fsdp["config"]["hidden"], ddp["config"]["hidden"])
        self.assertEqual(fsdp["config"]["batch_size_per_rank"],
                         ddp["config"]["batch_size_per_rank"])
        self.assertEqual(fsdp["environment"]["git_commit"],
                         ddp["environment"]["git_commit"])


if __name__ == "__main__":
    unittest.main()
