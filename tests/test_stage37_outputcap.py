import unittest
from pathlib import Path
from llm_opt_lab.serving.stage37_long_answer import main
from llm_opt_lab.serving.stage35_human_bench import frozen,prepare,SEEDS,MIXES
ROOT=Path(__file__).resolve().parents[1]
class LongAnswerHumanContracts(unittest.TestCase):
 def test_pinned_96_user_request_identites(self):
  rows,m,sys=frozen(ROOT/'data',ROOT/'results/stage35_public_data')
  for seed in SEEDS:
   for mix in ('cold0','hot50'):
    items=prepare(rows,seed,mix)
    self.assertEqual(len(items),96)
    self.assertEqual(len({x['request_id'] for x in items}),96)
    self.assertTrue(all(x['source_prompt'] for x in items))
  self.assertEqual(set(MIXES),{'cold0','hot50','hot100'})
