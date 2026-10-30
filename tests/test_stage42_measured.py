import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Stage42CompletedEvidenceTests(unittest.TestCase):
 def test_real_96_128_no_label_leak(self):
  folder=ROOT/'evidence/stage42_router'
  x=json.loads((folder/'frozen_router_result.json').read_text())
  self.assertEqual(x['status'],'measured')
  self.assertEqual(len(x['real_ifeval_generations']),96)
  self.assertEqual(len(x['real_ifeval_official_verdicts']),96)
  self.assertEqual(len(x['real_dolly_reference_ce']),128)
  self.assertEqual(x['previous_source_identites_excluded'],{'IFEval':325,'Dolly':960})
  self.assertFalse(x['router']['uses_official_labels_or_model_responses'])
  self.assertEqual(x['joint_gate_pass'],all(x['predeclared_joint_research_gate'].values()))
  self.assertEqual(sum(x['route_counts']['ifeval'].values()),96)
  self.assertEqual(sum(x['route_counts']['dolly'].values()),128)
  for row in x['real_dolly_reference_ce']:
   self.assertEqual(row['CE']['request_router'],row['CE'][row['route']])
 def test_sha256_manifest(self):
  folder=ROOT/'evidence/stage42_router'
  for line in (folder/'SHA256SUMS').read_text().splitlines():
   sha,name=line.split(None,1)
   self.assertEqual(sha,hashlib.sha256((folder/name.strip()).read_bytes()).hexdigest())
