import unittest
from pathlib import Path
from llm_opt_lab.serving.stage8_arrivals import frozen_requests
from llm_opt_lab.serving import stage22_stdlib_bench

ROOT=Path(__file__).resolve().parents[1]
class Stage22CorpusTests(unittest.TestCase):
 def test_frozen_request_identity_and_modes(self):
  b=stage22_stdlib_bench.bench
  corpus,manifest=frozen_requests(ROOT/'data/stage22_stdlib_requests.jsonl',ROOT/'data/stage22_stdlib_manifest.json')
  self.assertEqual(len(corpus),24)
  self.assertEqual(len({r['request_id'] for r in corpus}),24)
  self.assertEqual(len(manifest['modules']),24)
  self.assertEqual({r['bucket'] for r in corpus},{'short','medium','long','very_long'})
  self.assertTrue(all(1<=r['calibrated_prompt_tokens']['label']<=2048 for r in corpus))
  self.assertTrue(all(1<=r['calibrated_prompt_tokens']['detailed']<=2048 for r in corpus))
  for seed in (9301,9309):
   out=b.make_offers(corpus,seed)
   self.assertEqual(len(out),24)
   self.assertEqual(sum(x['output_mode']=='label' for x in out),12)
if __name__=='__main__':unittest.main()
