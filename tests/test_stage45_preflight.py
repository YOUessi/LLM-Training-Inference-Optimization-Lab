import unittest
from pathlib import Path
from llm_opt_lab.serving.stage45_router_load import COUNT,SEEDS,RATES,POLICIES,prepare,summarise
from llm_opt_lab.serving.stage35_human_bench import frozen
ROOT=Path(__file__).resolve().parents[1]
class Stage45DualEngineSourceTests(unittest.TestCase):
 def test_frozen_72_equal_24_each_and_stable_pairs(self):
  data,meta,_=frozen(ROOT/'data',ROOT/'results/stage35_public_data')
  for seed in SEEDS:
   x=prepare(data,seed)
   self.assertEqual(len(x),72)
   self.assertEqual(len({r['request_id'] for r in x}),72)
   for lang in ('en','es','zh'):
    self.assertEqual(sum(r['lang']==lang for r in x),24)
  self.assertEqual(set(POLICIES),{'base','lora_sft','request_router'})
  self.assertEqual(RATES,(4.,8.))
 def test_all_offered_denominator_is_72_not_96(self):
  rows=[{'request_id':f'case_{i}','status':'ok','ttft_from_due_ms':150.,
         'client_executor_wait_ms':0.,'completion_tokens':32} for i in range(72)]
  x=summarise(rows,12.)
  self.assertEqual(x['offered'],72)
  self.assertEqual(x['slo_all_offered']['250']['compliance_fraction'],1.)
  self.assertEqual(x['useful_output_tokens'],72*32)
