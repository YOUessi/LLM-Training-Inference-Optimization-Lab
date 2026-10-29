import json,hashlib,unittest
from pathlib import Path
from llm_opt_lab.serving.stage35_human_bench import frozen,prepare,SEEDS,MIXES
ROOT=Path(__file__).resolve().parents[1]
class Stage35HumanStudyTests(unittest.TestCase):
 def test_public_private_source_integrity_and_stratification(self):
  rows,m,system=frozen(ROOT/'data',ROOT/'results/stage35_public_data')
  self.assertEqual(len(rows),96)
  self.assertEqual(m['source_sha256'],'621ccd86a6ef320ca4e24c137121bd4b39bcc7a0df839f0897fcc965ef2076ed')
  self.assertTrue(m['raw_user_prompts_not_committed'])
  self.assertEqual(hashlib.sha256(system.encode()).hexdigest(),m['system_prompt_sha256'])
  for seed in SEEDS:
   prev=None
   for mix,hot in MIXES.items():
    x=prepare(rows,seed,mix)
    self.assertEqual(sum(r['prefix_state']=='hot' for r in x),hot)
    if mix=='hot50':
     self.assertEqual({l:sum(r['prefix_state']=='hot' and r['lang']==l for r in x) for l in ('en','es','zh')},
                     {'en':16,'es':16,'zh':16})
    if prev:self.assertEqual([r['request_id'] for r in prev],[r['request_id'] for r in x])
    prev=x
