"""CPU-only source + matched cold/hot experiment gates."""
import hashlib,json,unittest
from pathlib import Path
from llm_opt_lab.serving.stage30_mixed_bench import frozen_rows,prepare,summarise,N,SEEDS,MIXES
from llm_opt_lab.serving.stage8_arrivals import schedule
ROOT=Path(__file__).resolve().parents[1]
class Stage30Tests(unittest.TestCase):
 def test_unique_frozen_96_cold_hot_identity(self):
  rows,manifest=frozen_rows(ROOT/'data');self.assertEqual(len(rows),96)
  self.assertEqual(len({r['request_id'] for r in rows}),96)
  self.assertEqual(manifest['planned_mixes'],MIXES)
  self.assertEqual(manifest['source'].split(',')[0],'locally installed CPython PSF-licensed module/member documentation')
  for seed in SEEDS:
   previous=None
   for mix,k in MIXES.items():
    a=prepare(rows,seed,mix)
    self.assertEqual(sum(x['prefix_state']=='hot' for x in a),k)
    self.assertEqual(sum(x['output_mode']=='label' for x in a),48)
    self.assertTrue(all(x['target_prompt_tokens']<=1750 for x in a))
    if previous:self.assertEqual([x['request_id'] for x in previous],[x['request_id'] for x in a])
    previous=a
 def test_matched_openloop_offer_schedule(self):
  for seed in SEEDS:
   a=schedule(seed,8.,96,kind='fixed_window');b=schedule(seed,8.,96,kind='fixed_window')
   self.assertEqual(a,b);self.assertEqual(a[-1],95/8)
if __name__=='__main__':unittest.main()
