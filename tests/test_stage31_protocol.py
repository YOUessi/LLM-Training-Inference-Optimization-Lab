import json,hashlib,unittest
from pathlib import Path
from llm_opt_lab.training.stage26_real_sft import load_split
ROOT=Path(__file__).resolve().parents[1]
class FullDolly512SFTTests(unittest.TestCase):
 def test_frozen_two_epoch_split(self):
  meta=json.loads((ROOT/'data/stage26_manifest.json').read_text())
  self.assertEqual(meta['license'],'CC-BY-SA-3.0')
  for name,total in [('train',512),('validation',96),('test',96)]:
   blob=(ROOT/f'data/stage26_{name}.jsonl').read_bytes()
   self.assertEqual(hashlib.sha256(blob).hexdigest(),meta['splits'][name]['sha256'])
   self.assertEqual(len(blob.splitlines()),total)
 def test_train_config_contract(self):
  from llm_opt_lab.training.stage31_full_dolly import ALLOWED_SEEDS,ALLOWED_RANKS,STEPS_PER_EPOCH,EPOCHS
  self.assertEqual(ALLOWED_SEEDS,(3111,3127))
  self.assertEqual(ALLOWED_RANKS,(8,16))
  self.assertEqual(STEPS_PER_EPOCH*EPOCHS*2,1024)
