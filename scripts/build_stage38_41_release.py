"""Fail-closed fully evidence-backed Stage38–41 research report and rollout gates."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,statistics

def read(folder,file):return json.loads((pathlib.Path(folder)/file).read_text())

def verify(root:pathlib.Path):
 dirs={s:root/'evidence'/f'stage{s}_{suffix}' for s,suffix in [(38,'guarded_alpha'),(40,'constraint_rehearsal'),(41,'new_dolly128')]}
 for stage,folder in dirs.items():
  if not (folder/'SHA256SUMS').is_file():raise ValueError('unarchived Stage'+str(stage))
  for line in (folder/'SHA256SUMS').read_text().splitlines():
   digest,name=line.split(None,1)
   if hashlib.sha256((folder/name.strip()).read_bytes()).hexdigest()!=digest:
    raise ValueError('SHA mismatch Stage'+str(stage)+'/'+name)
 d38=read(dirs[38],'frozen_dev_selection.json');t38=read(dirs[38],'final_untouched101_test.json')
 d40=read(dirs[40],'dev_selection.json');t40=read(dirs[40],'final_test60_official.json')
 t41=read(dirs[41],'fresh128_four_conditions.json')
 if d38['status']!='dev_approved_for_holdout_not_production' or t38['status']!='measured':raise ValueError('Stage38 invalid')
 if d40['status']!='dev_frozen_no_final_test_access' or t40['status']!='measured':raise ValueError('Stage40 invalid')
 if t41['status']!='measured' or t41['counts']!={'examples':128,'categories':8,'models':4}:raise ValueError('Stage41 incomplete')
 if t38['dev_decision_sha256']!=hashlib.sha256((dirs[38]/'frozen_dev_selection.json').read_bytes()).hexdigest():raise ValueError('stage38 final selected after test')
 if t40['dev_selection_sha256']!=hashlib.sha256((dirs[40]/'dev_selection.json').read_bytes()).hexdigest():raise ValueError('stage40 final selected after test')
 if t41['selection38_file_sha256']!=hashlib.sha256((dirs[38]/'frozen_dev_selection.json').read_bytes()).hexdigest():raise ValueError('Dolly selected strength differs from earlier IFEval dev')
 if t41['selection40_file_sha256']!=hashlib.sha256((dirs[40]/'dev_selection.json').read_bytes()).hexdigest():raise ValueError('Dolly selected model differs from earlier IFEval dev')
 if d38['selected_alpha']!=t41['selection38_dev_alpha'] or d40['selected']!=t41['selection40_dev_model']:
  raise ValueError('test-driven reranking detected')
 return d38,t38,d40,t40,t41

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo',type=pathlib.Path,default=pathlib.Path('.'))
 p.add_argument('--report',type=pathlib.Path,default=pathlib.Path('docs/STAGE38_41_DELIVERY.md'))
 p.add_argument('--audit',type=pathlib.Path,default=pathlib.Path('evidence/stage38_41_research_release.json'))
 a=p.parse_args()
 d38,t38,d40,t40,t41=verify(a.repo)
 sel38=f'{d38["selected_alpha"]:.2f}'
 stats38=t38['scores']
 strict_base38=stats38['0.00']['strict_prompt_fraction'];strict_sel38=stats38[sel38]['strict_prompt_fraction']
 old38=stats38['1.00']['strict_prompt_fraction']
 sel40=d40['selected'];results40=t40['full_official_scores']
 b40=results40['base']['strict_prompt_fraction'];s40=results40[sel40]['strict_prompt_fraction'];old40=results40['old_full']['strict_prompt_fraction']
 quality41=t41['overall'];baseCE=quality41['base']['assistant_ce_nats'];scaleCE=quality41['guarded_strength']['assistant_ce_nats'];formatCE=quality41['format_replay']['assistant_ce_nats']
 # Deployment gate intentionally stricter than just passing a development subset.
 gate38={'source_and_selection_audited':True,'adapted_dev_feasible':d38['guardrail_passed_with_adaptation'],
  'unseen101_strict_not_below_base':strict_sel38>=strict_base38,
  'fresh128_Dolly_CE_improves_base':scaleCE<=baseCE-.1,
  'unseen101_CI_lower_not_below_minus_5pp':t38['paired_test_CIs_selected_minus']['base'][0]>=-.05}
 gate40={'source_and_selection_audited':True,'trained_dev_feasible':d40['selected_adapted'],
  'unseen60_strict_not_below_base':s40>=b40,
  'fresh128_Dolly_CE_improves_base':formatCE<=baseCE-.1,
  'unseen60_CI_lower_not_below_minus_5pp':t40['paired_stratified_CIs_selected_minus']['base'][0]>=-.05}
 research38=all(gate38.values());research40=all(gate40.values())
 status={'status':'evidence_verified_research_candidates_not_production_approved',
  'stage38':{'selected_alpha':d38['selected_alpha'],'dev_feasible':d38['guardrail_passed_with_adaptation'],
    'unseen_ifeval101':{'base_strict':strict_base38,'full_unscaled_strict':old38,'selected_strict':strict_sel38,
      'bootstrap95_selected_minus_base':t38['paired_test_CIs_selected_minus']['base']},
    'fresh128_dolly':{'base_ce':baseCE,'selected_ce':scaleCE},'candidate_gate':gate38,'research_candidate_pass':research38},
  'stage40':{'selected':sel40,'dev_feasible':d40['selected_adapted'],
    'unseen_ifeval60':{'base_strict':b40,'old_full_strict':old40,'selected_strict':s40,
      'bootstrap95_selected_minus_base':t40['paired_stratified_CIs_selected_minus']['base']},
    'fresh128_dolly':{'base_ce':baseCE,'selected_ce':formatCE},'candidate_gate':gate40,
    'research_candidate_pass':research40},
  'stage41':{'new_record_ids':128,'previous_selected_ids_excluded':832,'source':'Dolly15k CC BY-SA3.0',
    'all_four_answer_CE':{k:v['assistant_ce_nats'] for k,v in quality41.items()}},
  'production_rollout':{'allowed':False,'blocked_because':['Only public IFEval subsets, not full benchmark, and limited heldout samples',
   'No human review of factual/safety/instruction correctness on natural traffic',
   'Only one GPU and a limited model/training seed range; no multiGPU FSDP',
   'Prior system prefix cache wins conditional on engineered user prefix reuse and truncation']}}
 a.audit.parent.mkdir(parents=True,exist_ok=True);a.audit.write_text(json.dumps(status,indent=2)+'\n')
 lines=['# Stage38–41: real 8B training-instruction Pareto follow-up — final research audit',
  '','**Evidence audit PASSED for each distinct frozen input, dev selection, original adapter SHA and all completed model comparisons.** Model strengths and six constraint-rehearsal training models were selected only on separate dev sets; final IFEval source keys and new Dolly128 labels were never used for selecting the winner. **Production release remains BLOCKED** until independently verified real-user correctness and robust across-seed evaluation, regardless of any favorable per-task result.',
  '','## Stage38: inference-only LoRA strength and independent 101 IFEval test',
  '','| IFEval model | Strict obedience /101 | Strict rate |', '|---|---:|---:|',
  f"| Original Qwen3-8B (α0) | {stats38['0.00']['strict_met_count']}/101 | {strict_base38*100:.2f}% |",
  f"| Stage31 full rank16 (α1) | {stats38['1.00']['strict_met_count']}/101 | {old38*100:.2f}% |",
  f"| Frozen dev-selected α={d38['selected_alpha']:.2f} | {stats38[sel38]['strict_met_count']}/101 | {strict_sel38*100:.2f}% |",
  '',f"Dev43 adapted α feasible: {d38['guardrail_passed_with_adaptation']}. Independent test selected-minus-base bootstrap95 [{100*t38['paired_test_CIs_selected_minus']['base'][0]:+.2f},{100*t38['paired_test_CIs_selected_minus']['base'][1]:+.2f}] pp. Stage41 totally unused Dolly128 answer CE: base {baseCE:.4f}, selected strength {scaleCE:.4f}. Stage38 research-candidate strict go/no-go: **{'PASS' if research38 else 'BLOCKED'}**. The word-level repeat diagnostic from Stage39 is supplementary and must not be retroactively inserted into the dev-selection policy.",
  '', '## Stage40: six distinct real 8B deterministic constraint-rehearsal SFT runs + independent 60 IFEval test',
  '','| Model | Strict obedience /60 | Strict rate |', '|---|---:|---:|',
  f"| Base | {results40['base']['strict_met_count']}/60 | {b40*100:.2f}% |",
  f"| Previous Stage31 full naive SFT | {results40['old_full']['strict_met_count']}/60 | {old40*100:.2f}% |",
  f"| Dev-selected Stage40 `{sel40}` | {results40[sel40]['strict_met_count']}/60 | {s40*100:.2f}% |",
  '',f"Dev25 trained adaptation feasible: {d40['selected_adapted']}. Independent test selected-minus-base bootstrap95 [{100*t40['paired_stratified_CIs_selected_minus']['base'][0]:+.2f},{100*t40['paired_stratified_CIs_selected_minus']['base'][1]:+.2f}] pp. New Dolly128 answer CE: base {baseCE:.4f}, selected constraint SFT {formatCE:.4f}. Stage40 research-candidate strict go/no-go: **{'PASS' if research40 else 'BLOCKED'}**. All 6 source conditions used the same original 512 Dolly identities twice, with source-conditioned JSON/XML/markers or no reformatting; results are not human preference labels.",
  '', '## Stage41: 128 entirely new per-source identities, same Dolly upstream distribution',
  '| Frozen model | 128 original human-reference answer CE | Perplexity |','|---|---:|---:|']
 for model,values in quality41.items():
  lines.append(f"| {model} | {values['assistant_ce_nats']:.5f} | {values['perplexity']:.3f} |")
 lines+=['', '## Go/no-go and honest limits',
  '', 'Even when official prompt-level IFEval on *new source identities* improves, a small hand-filtered public source cannot prove general instruction adherence or correct human answers. The Stage38 dev43/test101 and Stage40 dev25/test60 source keys are disjoint from each other and from the 96 previously observed Stage36 keys, but all are from the same original public IFEval collection; the test may still overlap original Qwen3 pretraining. Stage41’s 128 new question identities are disjoint from 832 previous Dolly selections, but again from the same public dataset and possibly pretrained. 95% bootstrap intervals resample fixed test cases within small frozen tiers, not independent training seeds. No decisions from the final tests were fed back into the score selection. The presence of an adapted-model winner is reported by explicit guardrails, NOT automatically implied by lower Dolly CE.',
 '', '**Production deployment: BLOCKED.** Need real production load SLO + correct uncapped answers, human independent instruction/factual correctness, multi-dataset/seed generalization, and eventual multi-GPU training validation on multi-GPU hardware. GitHub code and raw stats remain the one source of truth, while nonselected large adapters are retained locally with SHA if necessary.',
 '', 'Full raw evidence: `docs/STAGE38_MEASURED_RESULTS.md`, `docs/STAGE40_MEASURED_RESULTS.md`, `docs/STAGE41_MEASURED_RESULTS.md`, `evidence/stage38_guarded_alpha`, `evidence/stage40_constraint_rehearsal`, `evidence/stage41_new_dolly128` and original IFEval Stage36 published negative result.']
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text('\n'.join(lines)+'\n')
 print('STAGE38_41_RELEASE_AUDIT',status['status'],'alpha',d38['selected_alpha'],'format',sel40,'passes',research38,research40)
if __name__=='__main__':main()
