import unittest,json
from pathlib import Path
from llm_opt_lab.serving.stage43_dual_adapter_smoke import MODELS
ROOT=Path(__file__).resolve().parents[1]
class DualModelApiPrerequisites(unittest.TestCase):
 def test_models_are_distinct_and_source_24_from_frozen_psf(self):
  self.assertEqual(MODELS['base'],'qwen3-8b-awq-local')
  self.assertEqual(MODELS['lora_sft'],'qwen3-8b-dolly-r16')
  self.assertNotEqual(MODELS['base'],MODELS['lora_sft'])
  rows=[json.loads(l) for l in (ROOT/'data/stage30_requests.jsonl').read_text().splitlines()]
  self.assertEqual(len(rows),96)
  self.assertEqual(len({x['request_id'] for x in rows[:24]}),24)
  self.assertTrue(all(x['prompts']['cold'] for x in rows[:24]))
 def test_only_loopback_server(self):
  txt=(ROOT/'scripts/run_stage43_vllm_lora_server.sh').read_text()
  self.assertIn('--host 127.0.0.1',txt)
  self.assertIn('--enable-lora',txt)
  self.assertIn('--max-lora-rank 16',txt)
  self.assertIn('--no-enable-prefix-caching',txt)
  self.assertNotIn('--host 0.0.0.0',txt)
