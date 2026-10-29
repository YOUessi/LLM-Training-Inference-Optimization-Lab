"""CI audit of the four final deliverable research-evidence families."""
import unittest
from pathlib import Path
from analysis.stage25_28_batch_acceptance import audit
ROOT=Path(__file__).resolve().parents[1]
class MultiTrackAcceptanceTests(unittest.TestCase):
 def test_four_research_evidence_families(self):
  result=audit(ROOT)
  self.assertEqual(result['status'],'research_artifact_evidence_pass_not_production_approved')
  for name in ('stage25_8b_serving','stage26_real_sft','stage27_qwen3_shapes','stage28_async_checkpoint'):
   self.assertEqual(result['workstreams'][name]['status'],'pass')
  self.assertGreater(len(result['open_critical_gates']),0)
if __name__=='__main__':unittest.main()
