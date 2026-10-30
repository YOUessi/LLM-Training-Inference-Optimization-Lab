import hashlib,html,json,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from llm_opt_lab.training.stage40_constraint_rehearsal import SEEDS,MIXES,RANK
from llm_opt_lab.training.stage40_policy_eval import corpus,MAX_DEV_LOSS_OF_STRICT,MIN_GAIN_DOLLY_CE
ROOT=Path(__file__).resolve().parents[1]
class Stage40SourceContracts(unittest.TestCase):
 def test_512_dolly_train_original_and_format_targets(self):
  manifest=json.loads((ROOT/'data/stage40_manifest.json').read_text())
  body=(ROOT/'data/stage40_dolly_format_replay.jsonl').read_bytes()
  self.assertEqual(hashlib.sha256(body).hexdigest(),manifest['dolly_format_replay_sha256'])
  rows=[json.loads(s) for s in body.decode().splitlines()]
  self.assertEqual(len(rows),512)
  self.assertEqual(len({x['identity_sha256'] for x in rows}),512)
  self.assertEqual(SEEDS,(4011,4027));self.assertEqual(MIXES,(0.,.25,.5));self.assertEqual(RANK,16)
  self.assertGreater(manifest['format_variant_valid_counts']['json_exact_valid'],500)
  for r in rows[:10]:
   for k,v in r['derived'].items():
    self.assertTrue(v['answer']);self.assertLessEqual(v['total_tokens'],384)
    if k=='json_exact':self.assertEqual(json.loads(v['answer']),{'answer':r['original_response']})
    if k=='xml_tag':self.assertEqual(ET.fromstring(v['answer']).text,r['original_response'])
    if k=='begin_end':
     self.assertTrue(v['answer'].startswith('START_ANSWER\n'))
     self.assertTrue(v['answer'].endswith('\nEND_ANSWER'))
 def test_unseen_stage40_ifeval_dev_test_disjoint_from_240_prior(self):
  dev,m=corpus(ROOT/'data','dev');test,_=corpus(ROOT/'data','test')
  self.assertEqual((len(dev),len(test)),(25,60))
  previous=[]
  for name in ('stage36_ifeval_96.jsonl','stage38_ifeval_dev.jsonl','stage38_ifeval_test.jsonl'):
   previous.extend(json.loads(s)['key'] for s in (ROOT/'data'/name).read_text().splitlines())
  self.assertEqual(len(set(previous)),240)
  self.assertFalse(set(previous)&({r['key'] for r in dev}|{r['key'] for r in test}))
  self.assertFalse({r['key'] for r in dev}&{r['key'] for r in test})
  self.assertTrue(m['no_eval_prompt_used_as_training_target'])
  self.assertEqual(MAX_DEV_LOSS_OF_STRICT,1)
  self.assertAlmostEqual(MIN_GAIN_DOLLY_CE,.1)
