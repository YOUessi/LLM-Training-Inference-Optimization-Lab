"""Real 6 QLoRA 1024-exposure cases and new IFEval60 official final checker."""
import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Stage40RealResultsTests(unittest.TestCase):
 def test_all_six_8b_sources_train_and_clean_test60(self):
  root=ROOT/'evidence/stage40_constraint_rehearsal'
  select=json.loads((root/'dev_selection.json').read_text())
  final=json.loads((root/'final_test60_official.json').read_text())
  self.assertEqual(select['status'],'dev_frozen_no_final_test_access')
  self.assertTrue(select['final_test_unaccessed'])
  self.assertEqual(len(select['training_candidates']),6)
  self.assertEqual(final['status'],'measured')
  self.assertTrue(final['dev25_disjoint'] and final['prior240_ifeval_keys_excluded'])
  self.assertEqual(final['dev_selection_sha256'],hashlib.sha256((root/'dev_selection.json').read_bytes()).hexdigest())
  self.assertEqual(final['selected'],select['selected'])
  for name in select['training_candidates']:
   a=json.loads((root/(name+'.json')).read_text())
   self.assertEqual(a['status'],'measured')
   self.assertEqual(a['config']['source_exposures'],1024)
   self.assertEqual(a['config']['optimizer_updates'],512)
   self.assertEqual(len(a['per_epoch']),2)
  for label,x in final['full_official_scores'].items():
   self.assertEqual(x['count'],60)
 def test_sha(self):
  root=ROOT/'evidence/stage40_constraint_rehearsal'
  for line in (root/'SHA256SUMS').read_text().splitlines():
   sha,name=line.split(None,1)
   self.assertEqual(sha,hashlib.sha256((root/name.strip()).read_bytes()).hexdigest())
