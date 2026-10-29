import json,unittest
from pathlib import Path
from analysis.stage30_real_analysis import analyze
ROOT=Path(__file__).resolve().parents[1]
class Stage30MeasuredTests(unittest.TestCase):
 def test_all_three_seeds_and_1728_real_offers(self):
  x=analyze(ROOT/'evidence/stage30_long_mixed')
  self.assertEqual(x['cases'],18)
  self.assertEqual(x['offered'],1728)
  self.assertEqual(x['completed'],1728)
  self.assertEqual(set(x['aggregate_by_mix']),{'cold0','hot50','hot100'})
  self.assertEqual(len(x['matched_pairs']),9)
  for mix,entry in x['aggregate_by_mix'].items():self.assertEqual(entry['paired_seed_count'],3)
