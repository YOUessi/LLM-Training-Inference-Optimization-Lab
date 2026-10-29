"""Dolly dataset provenance, exact nonleaking split, assistant-only labels."""
import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class DollySplits(unittest.TestCase):
 def test_licenses_hashes_and_no_leakage(self):
  d=json.loads((ROOT/'data/stage26_manifest.json').read_text())
  self.assertEqual(d['source_sha256'],'2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec')
  self.assertEqual(d['license'],'CC-BY-SA-3.0')
  identities=set()
  for split,n in [('train',512),('validation',96),('test',96)]:
   body=(ROOT/f'data/stage26_{split}.jsonl').read_bytes()
   self.assertEqual(hashlib.sha256(body).hexdigest(),d['splits'][split]['sha256'])
   rows=[json.loads(x) for x in body.decode().splitlines()]
   self.assertEqual(len(rows),n)
   self.assertEqual(len(set(x['identity_sha256'] for x in rows)),n)
   self.assertTrue(all(x['total_tokens']<=384 and x['target_tokens']>=2 for x in rows))
   self.assertEqual(len(set(x['category'] for x in rows)),8)
   self.assertFalse(set(x['identity_sha256'] for x in rows)&identities)
   identities.update(x['identity_sha256'] for x in rows)
 def test_response_only_mask_boundary(self):
  from transformers import AutoTokenizer
  from llm_opt_lab.training.stage26_real_sft import load_split
  t=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
  examples=load_split(ROOT/'data/stage26_validation.jsonl',t)
  self.assertEqual(len(examples),96)
  self.assertTrue(all(e['labels'][:len(e['input_ids'])-e['answer_count']]==[-100]*(len(e['input_ids'])-e['answer_count']) for e in examples))
  self.assertTrue(all(sum(z!=-100 for z in e['labels'])==e['answer_count'] for e in examples))
if __name__=='__main__':unittest.main()
