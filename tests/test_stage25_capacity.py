import hashlib,json,unittest
from pathlib import Path
from llm_opt_lab.serving.stage25_capacity_bench import DOMAINS
from llm_opt_lab.serving.stage8_arrivals import frozen_requests
from llm_opt_lab.serving.stage14_mixed_bench import make_offers
ROOT=Path(__file__).resolve().parents[1]
class FrozenStage25CorpusTests(unittest.TestCase):
 def test_two_distinct_source_cohorts_with_varying_context(self):
  corpus={}
  for name,(file,manifest,seeds) in DOMAINS.items():
   rows,meta=frozen_requests(ROOT/file,ROOT/manifest)
   self.assertEqual(len(rows),24)
   self.assertEqual(set(seeds),{9401,9409})
   self.assertEqual(meta['request_count'],24)
   self.assertEqual({r['bucket'] for r in rows},{'short','medium','long','very_long'})
   for r in rows:
    self.assertLessEqual(r['calibrated_prompt_tokens']['detailed']+256,2048)
    self.assertGreater(r['calibrated_prompt_tokens']['label'],0)
   self.assertEqual(len({r['request_id'] for r in rows}),24)
   self.assertEqual({len([r for r in rows if r['bucket']==bucket]) for bucket in ['short','medium','long','very_long']},{6})
   self.assertEqual(len(make_offers(rows,9401)),24)
   corpus[name]={r['source_module']:r['prompt_sha256'] for r in rows}
  self.assertEqual(set(corpus['shared']),set(corpus['distinct']))
  self.assertTrue(all(corpus['shared'][m]!=corpus['distinct'][m] for m in corpus['shared']))
