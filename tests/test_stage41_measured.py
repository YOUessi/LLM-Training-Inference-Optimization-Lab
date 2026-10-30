"""Real Dolly128 untouched records against four preselected model conditions."""
import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Stage41MeasuredTests(unittest.TestCase):
 def test_unseen_128_and_all_four_CE(self):
  root=ROOT/'evidence/stage41_new_dolly128'
  x=json.loads((root/'fresh128_four_conditions.json').read_text())
  self.assertEqual(x['status'],'measured')
  self.assertTrue(x['excluded_previous832_identites'])
  self.assertEqual(x['counts'],{'examples':128,'categories':8,'models':4})
  for name in ('base','original_full','guarded_strength','format_replay'):
   self.assertEqual(x['overall'][name]['examples'],128)
   self.assertEqual(len(x['per_example_answer_ce'][name]),128)
  for line in (root/'SHA256SUMS').read_text().splitlines():
   sha,name=line.split(None,1)
   self.assertEqual(sha,hashlib.sha256((root/name.strip()).read_bytes()).hexdigest())
