import json,hashlib,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Stage26ResultsTests(unittest.TestCase):
 def test_frozen_test_and_selected_adapter(self):
  root=ROOT/'evidence/stage26_sft_v2';data=json.loads((root/'analysis.json').read_text())
  self.assertEqual(data['status'],'pass');self.assertEqual(data['selected'],'B')
  self.assertEqual(data['test']['heldout_records'],96)
  self.assertLess(data['test']['selected_ce'],data['test']['base_ce'])
  selected=json.loads((root/'B.json').read_text())
  self.assertEqual(selected['config']['rank'],16)
  for filename,digest in selected['adapter_sha256'].items():
   self.assertEqual(hashlib.sha256((root/'selected_B_adapter'/filename).read_bytes()).hexdigest(),digest)
  self.assertEqual(json.loads((root/'selected_test.json').read_text())['test_metrics']['count'],96)
 def test_protocol_oom_evidence(self):
  x=(ROOT/'evidence/stage26_invalid_v1/C.log').read_text()
  self.assertIn('torch.OutOfMemoryError',x)
