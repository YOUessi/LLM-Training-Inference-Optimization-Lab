import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class AWQvsNF4ActualAnswerComparison(unittest.TestCase):
 def test_all_120_real_api_calls_and_original60_questions(self):
  p=ROOT/'evidence/stage46_awq_ifbench_quality'
  results=json.loads((p/'paired_AWQ_vs_NF4.json').read_text())
  scores=json.loads((p/'official_scores_public.json').read_text())
  self.assertEqual(results['status'],'verified')
  self.assertEqual(results['real_awq_api_calls'],120)
  self.assertEqual(results['original_ifbench_prompts'],60)
  self.assertEqual(len(results['per_prompt_comparison_no_text']),60)
  self.assertEqual(set(results['observed_model_registry']),{'qwen3-8b-awq-local','qwen3-8b-dolly-r16'})
  self.assertEqual(scores['kind'],'stage46_official_IFBench60_AWQ_vLLM_base_plus_LoRA')
  for m,stats in results['paired_roles'].items():
   self.assertEqual(sum(stats[k] for k in ('nf4_only_correct','awq_only_correct','both_correct','both_wrong')),60)
   self.assertEqual(stats['awq_strict_passes'],scores['aggregated'][m]['strict_correct'])
  self.assertTrue(results['conditioned_not_production_approval'])
 def test_hashes(self):
  p=ROOT/'evidence/stage46_awq_ifbench_quality'
  for line in (p/'SHA256SUMS').read_text().splitlines():
   sha,name=line.split(None,1)
   self.assertEqual(sha,hashlib.sha256((p/name.strip()).read_bytes()).hexdigest())
