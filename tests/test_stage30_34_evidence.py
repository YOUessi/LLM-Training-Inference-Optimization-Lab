"""Complete five-stage actual real GPU evidence release contract, not production approval."""
from pathlib import Path
import hashlib,json,unittest
from analysis.stage30_real_analysis import analyze as audit30
from analysis.stage33_sustained_analysis import analyze as audit33
ROOT=Path(__file__).resolve().parents[1]

def j(p):return json.loads(p.read_text())

class FiveWorkstreamMeasuredTests(unittest.TestCase):
 def test_30_and_33_real_serving_request_counts(self):
  a=audit30(ROOT/'evidence/stage30_long_mixed')
  b=audit33(ROOT/'evidence/stage33_sustained')
  self.assertEqual((a['cases'],a['completed']),(18,1728))
  self.assertEqual((b['real_cases'],b['completed']),(12,1152))
  self.assertEqual(a['aggregate_by_mix']['hot100']['paired_seed_count'],3)
  self.assertEqual(b['by_mix']['hot50']['paired_seed_count'],3)
 def test_31_full_sft_val_and_unique_test(self):
  root=ROOT/'evidence/stage31_full512';report=j(root/'analysis.json');selection=j(root/'selection.json')
  self.assertEqual(report['status'],'pass')
  self.assertEqual(report['training_exposures_per_model'],1024)
  self.assertEqual(len(report['candidates']),4)
  self.assertEqual(selection['selected'],'rank16_seed3127')
  self.assertEqual(selection['best_epoch'],1)
  self.assertEqual(report['heldout_test_records'],96)
  self.assertLess(report['frozen_test']['selected_assistant_ce'],report['frozen_test']['baseline_assistant_ce'])
 def test_32_semantic_proxy_and_negative_uncertainty(self):
  x=j(ROOT/'evidence/stage32_semantic/semantic_nli.json')
  self.assertEqual(x['status'],'measured')
  self.assertEqual(x['counts'],{'items':64,'models':3,'nli_directional_pairs':384})
  ci=x['stratified_bootstrap_95pct_ci_deltas_new_vs']['stage26_short']['mutual_entailment_mean']
  self.assertLess(ci[0],0)
  self.assertGreater(ci[1],0)
 def test_34_record_disjoint_from_704_and_ce_improvement(self):
  records=[json.loads(s) for s in (ROOT/'data/stage34_new_holdout.jsonl').read_text().splitlines()]
  prev=[]
  for split in ('train','validation','test'):
   prev += [json.loads(s)['identity_sha256'] for s in (ROOT/f'data/stage26_{split}.jsonl').read_text().splitlines()]
  self.assertEqual(len(set(prev)),704)
  self.assertEqual(len(records),128)
  self.assertFalse(set(prev)&{r['identity_sha256'] for r in records})
  x=j(ROOT/'evidence/stage34_unused128/fresh128_three_model_CE.json')
  self.assertLess(x['overall']['stage31_full']['ce_nats'],x['overall']['stage26_short']['ce_nats'])
 def test_all_archived_file_hashes(self):
  for stage in ('stage31_full512','stage32_semantic','stage33_sustained','stage34_unused128'):
   directory=ROOT/'evidence'/stage
   for line in (directory/'SHA256SUMS').read_text().splitlines():
    h,p=line.split(None,1)
    self.assertEqual(hashlib.sha256((directory/p.strip()).read_bytes()).hexdigest(),h)
