import unittest
from pathlib import Path
from analysis.stage25_real_analysis import analyze
ROOT=Path(__file__).resolve().parents[1]
class Stage25MeasuredEvidenceTests(unittest.TestCase):
 def test_full_528_offered_and_paired_source(self):
  x=analyze(ROOT/'evidence/stage25_measured')
  self.assertEqual(x['status'],'pass')
  self.assertEqual(x['cases'],22)
  self.assertEqual(x['offered_requests'],528)
  self.assertEqual(x['completed_requests'],528)
  self.assertEqual(x['factor_averages']['prefix_cache_shared']['paired_conditions'],8)
  self.assertEqual(x['factor_averages']['prefix_cache_distinct']['paired_conditions'],1)
if __name__=='__main__':unittest.main()
