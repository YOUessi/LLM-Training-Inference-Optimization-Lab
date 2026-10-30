import hashlib,json,unittest
from pathlib import Path
from llm_opt_lab.training.stage42_counterfactual import frozen,MODELS
ROOT=Path(__file__).resolve().parents[1]
class Stage42FrozenHoldoutTests(unittest.TestCase):
 def test_new_ifeval96_and_dolly128_are_disjoint(self):
  result,m=frozen(ROOT/'data')
  self.assertEqual(len(result['ifeval']),96)
  self.assertEqual(len(result['dolly']),128)
  self.assertEqual(MODELS,('base','lora_sft','request_router'))
  previous=set()
  for name in ('stage36_ifeval_96.jsonl','stage38_ifeval_dev.jsonl','stage38_ifeval_test.jsonl',
               'stage40_ifeval_dev.jsonl','stage40_ifeval_test.jsonl'):
   previous.update(json.loads(s)['key'] for s in (ROOT/'data'/name).read_text().splitlines())
  self.assertEqual(len(previous),325)
  self.assertFalse(previous&{x['key'] for x in result['ifeval']})
  original=set()
  for name in ('stage26_train.jsonl','stage26_validation.jsonl','stage26_test.jsonl',
               'stage34_new_holdout.jsonl','stage41_dolly_holdout128.jsonl'):
   original.update(json.loads(s)['identity_sha256'] for s in (ROOT/'data'/name).read_text().splitlines())
  self.assertEqual(len(original),960)
  self.assertFalse(original&{x['identity_sha256'] for x in result['dolly']})
  self.assertEqual(m['ifeval']['prior_325_excluded'],325)
  self.assertEqual(m['dolly']['prior_960_excluded'],960)
 def test_manifest_hashes_and_source_licences(self):
  _,m=frozen(ROOT/'data')
  self.assertEqual(m['source_ifeval_sha256'],'6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2')
  self.assertEqual(m['source_dolly_sha256'],'2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec')
  self.assertEqual(set(m['ifeval']['category_counts']),set(m['ifeval']['fixed_quota']))
  self.assertEqual(set(m['dolly']['category_counts'].values()),{16})
