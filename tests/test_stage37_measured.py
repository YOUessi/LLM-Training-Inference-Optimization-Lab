"""Audit exact 384-output stage37 measured server telemetry, counts, pairing and SHA."""
import hashlib,json,unittest
from pathlib import Path
from analysis.stage37_outputcap_analysis import analyze
ROOT=Path(__file__).resolve().parents[1]
class Stage37ActualGPUResults(unittest.TestCase):
 def test_12_real_cases_direct_prefix_hit_counters(self):
  a=analyze(ROOT/'evidence/stage37_long_answers',ROOT/'evidence/stage35_real_human')
  self.assertEqual(a['status'],'pass');self.assertEqual(a['cases'],12)
  self.assertEqual(a['offered'],1152);self.assertEqual(a['completed'],1152)
  self.assertEqual(len(a['paired_cache_conditions']),6)
  self.assertEqual(len(a['matched_cap128_vs384_hot50']),6)
  self.assertEqual(a['summary']['cold0']['cache_on_actual_prefix_hit_fraction']['mean'],0.)
  self.assertGreater(a['summary']['hot50']['cache_on_actual_prefix_hit_fraction']['mean'],.6)
  self.assertGreater(a['summary']['hot50']['on_hit_cap']['mean'],.7)
 def test_all_raw_sha256(self):
  folder=ROOT/'evidence/stage37_long_answers'
  for ln in (folder/'SHA256SUMS').read_text().splitlines():
   sha,name=ln.split(None,1)
   self.assertEqual(sha,hashlib.sha256((folder/name.strip()).read_bytes()).hexdigest())
