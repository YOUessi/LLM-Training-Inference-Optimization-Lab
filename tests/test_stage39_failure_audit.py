import json,unittest
from pathlib import Path
from analysis.stage39_ifeval_failure_audit import audit,exact_mcnemar,lexical_distinct2
ROOT=Path(__file__).resolve().parents[1]
class Stage39RealFailureAuditTests(unittest.TestCase):
 def test_paired_96_discordant_regression(self):
  r=audit(ROOT/'evidence/stage36_official_ifeval')
  self.assertEqual(r['status'],'verified_diagnostic_not_holdout')
  self.assertEqual(r['n'],96)
  self.assertEqual(r['model_stats']['base']['strict_count'],71)
  self.assertEqual(r['model_stats']['stage31_full']['strict_count'],52)
  self.assertEqual(r['model_stats']['stage31_full']['severely_repetitive_under_0_25_distinct2_count'],38)
  pair=r['base_comparisons']['base_vs_stage31_full']
  self.assertEqual((pair['base_only_correct'],pair['other_only_correct']),(24,5))
  self.assertLess(pair['exact_paired_McNemar_twosided'],0.001)
 def test_exact_binomial_and_diversity_bounded(self):
  self.assertEqual(exact_mcnemar(0,0),1.)
  self.assertEqual(exact_mcnemar(1,0),1.)
  self.assertLess(lexical_distinct2('a '*50),0.1)
  self.assertIsNone(lexical_distinct2('too short'))
