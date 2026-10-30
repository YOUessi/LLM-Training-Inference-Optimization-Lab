import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class IFBenchRealResultsTests(unittest.TestCase):
 def test_all_60_real_official_verifiers(self):
  f=ROOT/'evidence/stage44_external_ifbench/official_scores_public.json'
  x=json.loads(f.read_text())
  self.assertEqual(x['status'],'measured')
  self.assertEqual(x['count'],60)
  self.assertEqual(x['verifier_count'],60)
  self.assertEqual(len(x['public_anonymized_per_prompt_checks']),60)
  self.assertEqual(set(x['aggregated']),{'base','lora_sft','request_router'})
  self.assertEqual(x['cross_benchmark_go'],all(x['preregistered_generalization_acceptance_checks'].values()))
  self.assertEqual(len({r['key'] for r in x['public_anonymized_per_prompt_checks']}),60)
  self.assertNotIn('response',json.dumps(x['public_anonymized_per_prompt_checks']))
 def test_source_sha(self):
  folder=ROOT/'evidence/stage44_external_ifbench'
  for line in (folder/'SHA256SUMS').read_text().splitlines():
   sha,name=line.split(None,1)
   self.assertEqual(sha,hashlib.sha256((folder/name.strip()).read_bytes()).hexdigest())
