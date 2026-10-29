import json,hashlib,unittest
from pathlib import Path
from llm_opt_lab.training.stage32_semantic_quality import frozen_subset,lexical,MODELS
ROOT=Path(__file__).resolve().parents[1]
class SemanticStratificationTests(unittest.TestCase):
 def test_stable_64_distinct_and_eight_per_category(self):
  subset,manifest=frozen_subset(ROOT/'data')
  self.assertEqual(len(subset),64)
  self.assertEqual(len({r['identity_sha256'] for r in subset}),64)
  for c in sorted({r['category'] for r in subset}):
   self.assertEqual(sum(r['category']==c for r in subset),8)
  self.assertEqual(len(MODELS),3)
  self.assertEqual(manifest['splits']['test']['rows'],96)
 def test_lexical_metric_bounded(self):
  self.assertEqual(lexical('An animal is running','an animal is running')['word_f1'],1.)
  self.assertEqual(lexical('An animal is running','')['word_f1'],0.)
  self.assertEqual(lexical('a b c','c b a')['exact'],0)
if __name__=='__main__':unittest.main()
