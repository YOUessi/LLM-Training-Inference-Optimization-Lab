"""Release all measured 8B format-replay evidence, best adapter if admissible."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,shutil
SEEDS=(4011,4027);MIXES=(0.,.25,.5)

def ident(seed,mix):return f'mix{int(mix*100):02d}_seed{seed}'

def run(source:pathlib.Path,out:pathlib.Path):
 selection=json.loads((source/'dev_selection.json').read_text())
 final=json.loads((source/'final_test60_official.json').read_text())
 if selection['status']!='dev_frozen_no_final_test_access' or not selection['final_test_unaccessed']:
  raise ValueError('not a pre-frozen selection')
 if not(final['status']=='measured' and final['dev25_disjoint'] and final['prior240_ifeval_keys_excluded']):
  raise ValueError('not a clean untouched final test')
 if final['selected']!=selection['selected'] or final['dev_selection_sha256']!=hashlib.sha256((source/'dev_selection.json').read_bytes()).hexdigest():
  raise ValueError('model was reselected after final test')
 if set(selection['training_candidates'])!={ident(seed,mix) for seed in SEEDS for mix in MIXES}:
  raise ValueError('missing originally registered training candidates')
 training={}
 for seed in SEEDS:
  for mix in MIXES:
   key=ident(seed,mix);d=json.loads((source/(key+'.json')).read_text())
   if not(d['status']=='measured' and d['config']['source_exposures']==1024 and d['config']['optimizer_updates']==512):
    raise ValueError('invalid 2x512 training '+key)
   if len(d['per_epoch'])!=2 or any(x['unique_source_records']!=512 for x in d['per_epoch']):
    raise ValueError('incomplete source coverage '+key)
   if not(key in selection['metrics']):raise ValueError('training result absent from dev selection')
   training[key]=d
 out.mkdir(parents=True,exist_ok=True)
 for p in sorted(source.glob('*.json')):shutil.copy2(p,out/p.name)
 for p in sorted(source.glob('*.log')):shutil.copy2(p,out/p.name)
 for name in ('source_sha','progress.log'):
  f=source/name
  if f.is_file():shutil.copy2(f,out/name)
 binaries={}
 for name,record in training.items():
  adapter=source/(name+'_adapter')
  binaries[name]={}
  for filename,digest in record['adapter_sha256'].items():
   f=adapter/filename
   if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
    raise ValueError('trained adapter source SHA mismatch '+name+'/'+filename)
   binaries[name][filename]=digest
  if name==selection['selected']:
   folder=out/'selected_adapter';folder.mkdir(exist_ok=True)
   for f in adapter.iterdir():
    if f.is_file():shutil.copy2(f,folder/f.name)
 (out/'all_six_adapter_sha256.json').write_text(json.dumps({'all_six_have_local_verified_blobs':True,
     'only_selected_adapter_published_if_nonbase':selection['selected']!='base',
     'original_artifacts_retained_on_tang':str(source),'by_candidate':binaries},indent=2)+'\n')
 (out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(out))+'\n'
    for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
 lines=['# Stage40: real 8B validated constraint-format replay SFT vs original instruction obedience',
  '', '**Actual measurements** of six Qwen3-8B NF4 QLoRA rank16 runs on one RTX4090 Laptop. Each run used 512 original Dolly CC BY-SA3.0 human-source identities twice (1,024 exposed source records and 512 optimizer updates), with fixed 0/25/50%-format augmentation and two independent train/data seeds 4011/4027. JSON/XML/begin-end constraint targets were algorithmically and deterministically transformed from human answers (not human-labeled new tasks). All training data and format verifiers had been frozen before these results. No IFEval prompt used as a training target. All six raw metrics and training logs archived in `evidence/stage40_constraint_rehearsal/`.',
  '', '## Dev25: selection based on BOTH official instruction obedience and Dolly validation CE',
  '', '| Candidate | Format share | Dolly96 validation CE | IFEval dev25 strict pass count | Output cap hits /25 | Dev guarded eligible |',
  '|---|---:|---:|---:|---:|---|']
 for label,stats in selection['metrics'].items():
  mix=training[label]['config']['mix'] if label in training else '-'
  passes=all(stats.get('guardrails',{}).values()) if label in training else False
  lines.append(f"| {label} | {mix} | {stats['Dolly_validation_ce']:.4f} | {stats['strict']}/25 | {stats['capped']}/25 | {'YES' if passes else 'NO/reference'} |")
 lines+=['',f"**Pre-frozen model choice:** `{selection['selected']}`. Feasible constraint-trained adapters (before final test): {selection['feasible_adapted']}. Promotion constraints were: strict official dev compliance no more than one question below original base, at most two extra 384-token truncations, and Dolly validation CE at least 0.10 nats/token lower than base. If none feasible choose original base; this is a normal negative outcome, not a reason to invent a winner.",
  '', '## Independent final Google IFEval60: no model choice after observing final test',
  '', '| Frozen model | Official strict pass /60 | Strict all-instruction prompt accuracy | Loose prompt accuracy | Reached 384 output cap |',
  '|---|---:|---:|---:|---:|']
 for k,score in final['full_official_scores'].items():
  lines.append(f"| {k} | {score['strict_met_count']}/60 | {score['strict_prompt_fraction']*100:.1f}% | {score['loose_prompt_fraction']*100:.1f}% | {score['hit_cap_fraction']*100:.1f}% |")
 lines+=['', '### Exploratory bootstrap CI for selected minus comparator strict success rate',
  '| Comparator | Selected−comparator within-family fixed-test 95% CI |','|---|---:|']
 for comparator,bounds in final['paired_stratified_CIs_selected_minus'].items():
  lines.append(f'| {comparator} | [{bounds[0]*100:+.1f}, {bounds[1]*100:+.1f}] percentage points |')
 lines+=['', '**Interpretation and limits:** Real format demonstrations do not guarantee improved rule-following. The independent test contains only **60 public IFEval prompts from five categories**, selected from remaining keys after excluding all Stage36/38 observed prompts (240 keys). Not full 541 IFEval, not a natural production user population. Two training seeds per replay share cannot establish robust generality. Training-data transformation into JSON/XML/markers is deterministic but not independently human reviewed. Original model could have seen public Dolly/IFEval during pretraining; exact source and test hashes remain in the manifest.',
  '', 'Only validation-selected best trained adapter is published, if any passes the pre-frozen rule; all other five original adapter files remain on Tang with their verified Git blob hashes and fully reproducing train source/config/logs. No unsupported claim that a test-driven model is safe, semantically correct, human-aligned or suitable for rollout.']
 pathlib.Path('docs/STAGE40_MEASURED_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print('STAGE40_ARCHIVED',len(training),'trained cases','selected',selection['selected'],'feasible',len(selection['feasible_adapted']))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=pathlib.Path,required=True)
 p.add_argument('--out',type=pathlib.Path,default=pathlib.Path('evidence/stage40_constraint_rehearsal'))
 a=p.parse_args();run(a.input,a.out)
