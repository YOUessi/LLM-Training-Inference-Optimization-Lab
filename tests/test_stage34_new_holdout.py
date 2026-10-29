import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class NewHoldoutTests(unittest.TestCase):
 def test_128_untouched_dolly_records(self):
  data=(ROOT/'data/stage34_new_holdout.jsonl').read_bytes()
  m=json.loads((ROOT/'data/stage34_manifest.json').read_text())
  self.assertEqual(m['status'],'frozen')
  self.assertEqual(m['source_original_sha256'],'2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec')
  self.assertEqual(hashlib.sha256(data).hexdigest(),m['input_sha256'])
  rows=[json.loads(s) for s in data.decode().splitlines()]
  self.assertEqual(len(rows),128)
  self.assertEqual(len({r['identity_sha256'] for r in rows}),128)
  prev=set()
  for split in ('train','validation','test'):
   prev.update(json.loads(s)['identity_sha256'] for s in (ROOT/f'data/stage26_{split}.jsonl').read_text().splitlines())
  self.assertEqual(len(prev),704)
  self.assertFalse(prev & {r['identity_sha256'] for r in rows})
  self.assertEqual(set(m['category_counts'].values()),{16})
