"""Materialize Stage38 dev43/holdout101 results without changing pre-selected alpha."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,shutil

def run(root:pathlib.Path,out:pathlib.Path):
 dev=json.loads((root/'frozen_dev_selection.json').read_text())
 test=json.loads((root/'final_untouched101_test.json').read_text())
 if not(dev['status']=='dev_approved_for_holdout_not_production' and
        dev['new_dev_count']==43 and dev['holdout_count']==101 and
        test['status']=='measured' and test['new_holdout_count']==101 and test['dev43_disjoint']):
  raise ValueError('invalid staged development/final result')
 if dev['selected_key'] not in dev['metrics'] or dev['selected_alpha']!=test['selected_alpha']:
  raise ValueError('test selection changed')
 if test['dev_decision_sha256']!=hashlib.sha256((root/'frozen_dev_selection.json').read_bytes()).hexdigest():
  raise ValueError('test not locked to pre-test dev decision')
 selected=dev['selected_key']
 if bool(dev['feasible_scales'])!=bool(dev['guardrail_passed_with_adaptation']):raise ValueError('feasibility changed')
 if dev['guardrail_passed_with_adaptation']:
  scores={k:v for k,v in dev['metrics'].items() if k!='0.00' and all(v['guardrail_checks'].values())}
  if not scores or min(scores,key=lambda k:(scores[k]['dolly_val']['ce_nats'],scores[k]['alpha']))!=selected:
   raise ValueError('dev scoring protocol not observed')
 else:
  if selected!='0.00':raise ValueError('nonfeasible model promoted')
 for key,actual in test['scores'].items():
  if actual['count']!=101 or len(actual['per_case'])!=101:raise ValueError('final full test missing '+key)
 out.mkdir(parents=True,exist_ok=True)
 for p in sorted(root.glob('*.json')):
  shutil.copy2(p,out/p.name)
 for fn in ('source_sha','dev.log','test.log'):
  if (root/fn).is_file():shutil.copy2(root/fn,out/fn)
 # Secondary post-hoc decoding degeneration diagnostic; NEVER modify dev selection.
 from analysis.stage39_ifeval_failure_audit import lexical_distinct2
 raw=json.loads((root/'final_101_outputs.json').read_text())
 repetition={}
 for key,records in raw['generation_by_scale'].items():
  if len(records)!=101:raise ValueError('not all holdout generations for '+key)
  counts=[lexical_distinct2(x['response']) for x in records]
  repetition[key]={'count':len(records),
   'severe_distinct2_below0_25_count':sum(v is not None and v<.25 for v in counts),
   'mean_distinct2':sum(v for v in counts if v is not None)/max(1,sum(v is not None for v in counts))}
 (out/'repetition_posthoc.json').write_text(json.dumps({'posthoc_not_used_for_selection':True,
    'metric':'word bigram diversity, threshold 0.25, language sensitive',
    'by_scale':repetition},indent=2)+'\n')
 (out/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n'
  for f in sorted(out.glob('*')) if f.is_file() and f.name!='SHA256SUMS'))
 source=dev['source_sha'][:8]
 lines=['# Stage38: guarded adapter-strength recovery of official IFEval obedience',
  '', '**Measured** real Qwen3-8B NF4 Stage31 rank16 LoRA. Strictly excluded **all 96 previously observed Stage36** IFEval keys. Preregistered **43 new development instructions** used to select inference-time adapter residual strength from 0/.25/.5/.75/1.0 based on BOTH Google official strict compliance and Dolly val96 answer-only CE/cap-hit guardrails. Final **101 different untouched IFEval source instructions** used exactly once after `frozen_dev_selection.json` exists. No parameter training or test-driven strength choice in Stage38. Output context384 greedy no-thinking. All source and adapter SHA checks, official grader per-case verdicts and actual generated outputs are archived in `evidence/stage38_guarded_alpha/`.',
  '', '## Development-only Pareto preselection (not independent final result)',
  '', '| LoRA residual strength | Official strict passed /43 | Dolly validation CE | Outputs at384/43 | All predeclared constraints passed |',
  '|---|---:|---:|---:|---|']
 for k,m in sorted(dev['metrics'].items(),key=lambda kv:kv[1]['alpha']):
  passed=all(m.get('guardrail_checks',{}).values()) if k!='0.00' else False
  lines.append(f"| {m['alpha']:.2f} | {m['ifeval_dev']['strict_met_count']}/43 | {m['dolly_val']['ce_nats']:.5f} | {m['ifeval_dev']['hit_cap_count']}/43 | {'YES' if passed else 'NO/baseline'} |")
 lines+=['',f"**Frozen selected α = {dev['selected_alpha']:.2f}**; genuinely adapted model satisfies all three dev guards: **{dev['guardrail_passed_with_adaptation']}**. Guardrails were: dev strict success at least unadapted base−2, cap-hits≤base+2, and Dolly val answer CE reduction≥0.10. If none eligible, base α0 is automatically selected (no adapted win). No real IFEval final-test answer consulted by this selection.",
  '', '## Final untouched IFEval101 official strict/loose test',
  '', '| Fixed comparison model | Strict passed /101 | Strict prompt accuracy | Loose prompt accuracy | Cap384 hit fraction |',
  '|---|---:|---:|---:|---:|']
 for key,stats in sorted(test['scores'].items(),key=lambda kv:float(kv[0])):
  lines.append(f"| α={key} | {stats['strict_met_count']}/101 | {stats['strict_prompt_fraction']*100:.2f}% | {stats['loose_prompt_fraction']*100:.2f}% | {stats['hit_cap_fraction']*100:.1f}% |")
 lines+=['', '### Predeclared selected minus comparators: within-fixed-test 95% stratified bootstrap interval (exploratory)',
 '', '| Comparator | Selected−comparator strict accuracy interval |','|---|---:|']
 for alias,ci in test['paired_test_CIs_selected_minus'].items():
  lines.append(f'| {alias} | [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] percentage points |')
 lines+=['', '## Post-hoc repetition diagnostic (not an additional tuning rule)', '', '| Scale | Low distinct-2 /101 (word bigram heuristic) | Mean distinct-2 |', '|---|---:|---:|']
 for key,stat in repetition.items():
  lines.append(f"| {key} | {stat['severe_distinct2_below0_25_count']}/101 | {stat['mean_distinct2']:.3f} |")
 lines+=['', '**Judgment:** Higher Dolly reference CE performance alone is not an alignment result; preserve exact official IFEval prompt/constraint results, per-family failures, and 384-token caps. One set of 101 public IFEval source questions and one trained LoRA checkpoint do not establish production user safety, semantic correctness, or multi-seed robustness. Residual scaling is a legitimate inference-time Pareto control but not a newly trained model/architecture. Full source license Apache 2.0 and Dolly validation license CC BY-SA3.0. All raw development and test outputs are available in repo. The earlier 96 IFEval results informed the *hypothesis* and are not counted as independent verification.']
 pathlib.Path('docs/STAGE38_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE38_ARCHIVED_SOURCE',source,'selected',selected,'valid_untouched_tests',len(test['scores']),'pass',dev['guardrail_passed_with_adaptation'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage38_guarded_alpha'));a=p.parse_args()
 run(a.input,a.out)
