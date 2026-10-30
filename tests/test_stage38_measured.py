"""Real GPU locked dev43/untouched test101 from Stage38."""
import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Stage38RealResultsTests(unittest.TestCase):
 def test_frozen_dev_guard_then_untouched_101(self):
  folder=ROOT/'evidence/stage38_guarded_alpha'
  dev=json.loads((folder/'frozen_dev_selection.json').read_text())
  test=json.loads((folder/'final_untouched101_test.json').read_text())
  self.assertEqual(dev['status'],'dev_approved_for_holdout_not_production')
  self.assertEqual((dev['new_dev_count'],dev['holdout_count']),(43,101))
  self.assertEqual(test['status'],'measured')
  self.assertEqual(test['selected_alpha'],dev['selected_alpha'])
  self.assertEqual(test['dev_decision_sha256'],hashlib.sha256((folder/'frozen_dev_selection.json').read_bytes()).hexdigest())
  for key,scores in test['scores'].items():
   self.assertEqual(scores['count'],101)
   self.assertEqual(len(scores['per_case']),101)
  self.assertEqual(dev['selected_key'],'0.00' if not dev['feasible_scales'] else
      min(dev['feasible_scales'],key=lambda k:(dev['metrics'][k]['dolly_val']['ce_nats'],dev['metrics'][k]['alpha'])))
 def test_archive_hash_manifest(self):
  folder=ROOT/'evidence/stage38_guarded_alpha'
  for line in (folder/'SHA256SUMS').read_text().splitlines():
   sha,name=line.split(None,1)
   self.assertEqual(sha,hashlib.sha256((folder/name.strip()).read_bytes()).hexdigest())
