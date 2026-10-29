import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'evidence/stage28_measured'
class Stage28EvidenceTests(unittest.TestCase):
 def test_full_train_exact_recovery(self):
  d=json.loads((ROOT/'comparison.json').read_text())
  self.assertEqual(d['status'],'pass')
  self.assertTrue(d['exact_final_adapter_and_optimizer'])
  self.assertEqual(len(d['runs']),4)
  self.assertEqual(len(d['checkpoints_verified']),10)
  self.assertEqual(len({x['adapter_hash'] for x in d['runs']}),1)
  self.assertEqual(len({x['optimizer_hash'] for x in d['runs']}),1)
  self.assertEqual(d['runs'][-1]['restarted_from_step'],6)
 def test_published_representatives_integrity(self):
  meta=json.loads((ROOT/'full_checkpoints_manifest.json').read_text())
  self.assertEqual(len(meta['all_checkpoints']),10)
  for path in meta['published_representatives']:
   info=next(x for x in meta['all_checkpoints'] if x['path']==path)
   self.assertEqual(hashlib.sha256((ROOT/'checkpoints'/path).read_bytes()).hexdigest(),info['sha256'])
