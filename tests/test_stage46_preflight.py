import unittest
from pathlib import Path
from llm_opt_lab.serving.stage46_vllm_ifeval_quality import MODEL_IDS,BASE_URL
ROOT=Path(__file__).resolve().parents[1]
class AWQIFBenchRealDeployPreflight(unittest.TestCase):
 def test_dual_models_fixed_and_vllm_loopback_only(self):
  self.assertEqual(MODEL_IDS,{'base':'qwen3-8b-awq-local','lora_sft':'qwen3-8b-dolly-r16'})
  self.assertEqual(BASE_URL,'http://127.0.0.1:8033')
  code=(ROOT/'scripts/run_stage46_awq_ifbench.sh').read_text()
  self.assertIn('flock -n 9',code)
  self.assertIn('stage44_ifbench_external60.jsonl',code)
  self.assertIn('run_stage43_vllm_lora_server.sh',code)
  self.assertIn('private_generation.json',code)
  self.assertIn('official_scores.json',code)
