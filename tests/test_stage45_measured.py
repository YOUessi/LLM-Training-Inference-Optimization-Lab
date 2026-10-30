import hashlib,json,unittest
from pathlib import Path
from analysis.stage45_dual_load_analysis import analyze
ROOT=Path(__file__).resolve().parents[1]
class Stage45RealConcurrentDualEngineTests(unittest.TestCase):
 def test_all_864_success_and_true_model_ids(self):
  r=analyze(ROOT/'evidence/stage45_dual_vllm_load')
  self.assertEqual(r['status'],'pass')
  self.assertEqual(r['actual_cells'],12)
  self.assertEqual(r['actual_offered'],864)
  self.assertEqual(r['actual_completed'],864)
  self.assertEqual(len(r['paired_2seeds_2rates']),4)
  self.assertEqual(set(r['by_rate']),{'4','8'})
 def test_sha256(self):
  folder=ROOT/'evidence/stage45_dual_vllm_load'
  for line in (folder/'SHA256SUMS').read_text().splitlines():
   sha,name=line.split(None,1)
   self.assertEqual(sha,hashlib.sha256((folder/name.strip()).read_bytes()).hexdigest())
