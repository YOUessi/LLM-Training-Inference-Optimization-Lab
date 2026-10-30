import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Stage43MeasuredVllm(unittest.TestCase):
 def test_two_real_model_ids_and_72_responses(self):
  root=ROOT/'evidence/stage43_dual_vllm'
  x=json.loads((root/'real_dual_model_results.json').read_text())
  self.assertEqual(x['status'],'measured')
  self.assertEqual(len(x['request_results']),72)
  self.assertEqual(set(x['registered_model_ids']),set(x['model_ids'].values()))
  for mode in ('base','lora_sft','router'):
   self.assertEqual(x['stats'][mode]['completed'],24)
   self.assertGreater(x['stats'][mode]['mean_ttft_ms'],0)
 def test_pinned_manifest(self):
  root=ROOT/'evidence/stage43_dual_vllm'
  for line in (root/'SHA256SUMS').read_text().splitlines():
   sha,name=line.split(None,1)
   self.assertEqual(sha,hashlib.sha256((root/name.strip()).read_bytes()).hexdigest())
