"""Verify four honest completed Qwen3 real-GPU follow-up evidence families."""
import hashlib,json,unittest
from pathlib import Path
from analysis.stage33_sustained_analysis import analyze as stage33_audit
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text())

class Stage31To34ActualEvidenceTests(unittest.TestCase):
 def test_full512_two_epoch_four_models_and_selected_blob(self):
  root=ROOT/'evidence/stage31_full512'
  summary=read(root/'analysis.json');sel=read(root/'selection.json');t=read(root/'selected_final_test.json')
  self.assertEqual(summary['status'],'pass')
  self.assertEqual(summary['training_exposures_per_model'],1024)
  self.assertEqual(len(summary['candidates']),4)
  self.assertEqual(sel['selected'],'rank16_seed3127')
  self.assertEqual(sel['best_epoch'],1)
  for name,record in summary['candidates'].items():
   self.assertEqual(record['train_exposures'],1024)
   self.assertEqual(record['best_epoch'],1)
   self.assertLess(record['epoch1_validation_ce'],record['epoch2_validation_ce'])
  self.assertEqual(t['test_metrics']['count'],96)
  self.assertLess(t['test_metrics']['assistant_ce_nats'],t['base_test_metrics']['assistant_ce_nats'])
  selected=read(root/(sel['selected']+'.json'))
  for filename,digest in selected['adapter_sha256'].items():
   p=root/'selected_adapter'/filename
   self.assertTrue(p.is_file())
   self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),digest)
 def test_real_independent_nli_semantic_proxy(self):
  root=ROOT/'evidence/stage32_semantic'
  generations=read(root/'three_model_generations.json')
  nli=read(root/'semantic_nli.json')
  self.assertEqual(generations['status'],'measured')
  self.assertEqual(len(generations['cases']),64)
  self.assertEqual(nli['counts'],{'items':64,'models':3,'nli_directional_pairs':384})
  self.assertEqual(nli['model_generations_sha256'],hashlib.sha256((root/'three_model_generations.json').read_bytes()).hexdigest())
  self.assertTrue(nli['stratified_bootstrap_95pct_ci_deltas_new_vs']['stage26_short']['mutual_entailment_mean'][0]<0)
  self.assertTrue(nli['stratified_bootstrap_95pct_ci_deltas_new_vs']['stage26_short']['mutual_entailment_mean'][1]>0)
 def test_sustained_12_real_pairs(self):
  audit=stage33_audit(ROOT/'evidence/stage33_sustained')
  self.assertEqual(audit['status'],'pass')
  self.assertEqual(audit['offered'],1152)
  self.assertEqual(audit['real_cases'],12)
  self.assertEqual(audit['by_mix']['hot50']['paired_seed_count'],3)
 def test_new128_never_before_selected(self):
  x=read(ROOT/'evidence/stage34_unused128/fresh128_three_model_CE.json')
  m=read(ROOT/'data/stage34_manifest.json')
  self.assertEqual(x['status'],'measured')
  self.assertEqual(x['counts']['examples'],128)
  self.assertEqual(x['frozen_unused_holdout_sha256'],m['input_sha256'])
  prior=set()
  for split in ('train','validation','test'):
   prior|={json.loads(s)['identity_sha256'] for s in (ROOT/f'data/stage26_{split}.jsonl').read_text().splitlines()}
  current={json.loads(s)['identity_sha256'] for s in (ROOT/'data/stage34_new_holdout.jsonl').read_text().splitlines()}
  self.assertEqual(len(prior),704)
  self.assertEqual(len(current),128)
  self.assertFalse(prior&current)
  self.assertLess(x['overall']['stage31_full']['ce_nats'],x['overall']['stage26_short']['ce_nats'])
  self.assertLess(x['overall']['stage26_short']['ce_nats'],x['overall']['base']['ce_nats'])
 def test_archived_evidence_hashes(self):
  for folder in ('stage31_full512','stage32_semantic','stage33_sustained','stage34_unused128'):
   root=ROOT/'evidence'/folder
   checks=(root/'SHA256SUMS').read_text().splitlines()
   self.assertGreater(len(checks),2)
   for line in checks:
    digest,name=line.split(None,1)
    self.assertEqual(hashlib.sha256((root/name.strip()).read_bytes()).hexdigest(),digest)
