import unittest
from pathlib import Path
from llm_opt_lab.training.stage41_new128_tradeoff import audit_sources
ROOT=Path(__file__).resolve().parents[1]
class Stage41UntouchedRecordTests(unittest.TestCase):
 def test_frozen_disjoint128_after_prior832(self):
  manifest=audit_sources(ROOT/'data')
  self.assertEqual(manifest['prior832_excluded_count'],832)
  self.assertEqual(manifest['records'],128)
  self.assertEqual(set(manifest['category_counts'].values()),{16})
  self.assertEqual(manifest['source_sha256'],'2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec')
