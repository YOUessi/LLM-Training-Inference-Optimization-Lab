import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'evidence/stage29_measured'
class Stage29EvidenceTests(unittest.TestCase):
 def test_all_96_holdout_pairs_and_hash(self):
  q=json.loads((ROOT/'quality.json').read_text())
  self.assertEqual(q['status'],'measured')
  self.assertEqual(q['total_cases'],96)
  self.assertEqual(len(q['cases']),96)
  self.assertEqual(len({x['id_sha256'] for x in q['cases']}),96)
  self.assertEqual(q['generation']['enable_thinking'],False)
  self.assertGreater(q['overall']['paired_diff']['rouge_l_f1'],0)
  self.assertEqual(q['overall']['tuned_better_rouge_cases']+q['overall']['base_better_rouge_cases'],91)
  for line in (ROOT/'SHA256SUMS').read_text().splitlines():
   sha,name=line.split(None,1)
   self.assertEqual(sha,hashlib.sha256((ROOT/name.strip()).read_bytes()).hexdigest())
if __name__=='__main__':unittest.main()
