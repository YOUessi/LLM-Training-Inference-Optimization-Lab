"""Audit four fully trained 8B QLoRA models, pure validation selection and test once."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path
NAMES=('rank8_seed3111','rank16_seed3111','rank8_seed3127','rank16_seed3127')

def analyze(root:Path,data:Path)->dict:
 manifest=json.loads((data/'stage26_manifest.json').read_text())
 selection=json.loads((root/'selection.json').read_text())
 if selection['status']!='candidate_selected_from_validation_only' or not selection['test_has_not_been_accessed']:
  raise ValueError('invalid frozen selection protocol')
 files={name:json.loads((root/(name+'.json')).read_text()) for name in NAMES}
 source=(root/'source_sha').read_text().strip()
 if len(source)!=40:raise ValueError('missing pinned git source')
 summaries={}
 for name,r in files.items():
  if r['status']!='measured' or r['train_unique_records']!=512 or r['train_exposures']!=1024:
   raise ValueError('not exactly two full training epochs '+name)
  if len(r['train_epochs'])!=2 or r['config']['total_updates']!=512:
   raise ValueError('incomplete 512 update training '+name)
  if r['dataset_sha256']!=manifest['source_sha256']:
   raise ValueError('upstream license/corpus source mismatch')
  if r['source_splits']!={s:v['sha256'] for s,v in manifest['splits'].items()}:
   raise ValueError('split drift '+name)
  if not all(e['train_records_seen_unique']==512 and e['validation']['count']==96 for e in r['train_epochs']):
   raise ValueError('bad train coverage/validation '+name)
  if r['selected_by_validation_epoch'] not in (1,2):raise ValueError('missing best epoch')
  actual=min(e['validation']['assistant_ce_nats'] for e in r['train_epochs'])
  if not math.isclose(actual,r['best_validation_ce'],rel_tol=0,abs_tol=1e-8):raise ValueError('incorrect best checkpoint')
  for filename,sha in r['adapter_sha256'].items():
   p=root/(name+'_adapter')/filename
   if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
    raise ValueError('adapter digest mismatch '+name+'/'+filename)
  summaries[name]={'config':r['config'],'train_exposures':r['train_exposures'],
    'epoch1_validation_ce':r['train_epochs'][0]['validation']['assistant_ce_nats'],
    'epoch2_validation_ce':r['train_epochs'][1]['validation']['assistant_ce_nats'],
    'best_epoch':r['selected_by_validation_epoch'],'best_validation_ce':r['best_validation_ce'],
    'samples_per_second':r['training_efficiency']['samples_per_second_excluding_first5'],
    'peak_allocated_gib':r['training_efficiency']['peak_allocated_gib'],
    'trainable_parameters':r['trainable_parameters'],
    'selected_checkpoint_shas':r['adapter_sha256']}
 actual=min(files,key=lambda n:files[n]['best_validation_ce'])
 if actual!=selection['selected']:raise ValueError('test leakage: selection differs from min validation')
 test=json.loads((root/'selected_final_test.json').read_text())
 if test['status']!='measured' or test['action']!='test' or test['test_metrics']['count']!=test['base_test_metrics']['count']!=96:
  raise ValueError('missing complete independent 96 test')
 if test['config']['rank']!=selection['rank'] or test['dataset_sha256']!=manifest['source_sha256']:
  raise ValueError('test does not use selected adapter configuration')
 if test['splits']!={s:v['sha256'] for s,v in manifest['splits'].items()}:
  raise ValueError('test split changed')
 paired=[]
 for seed in (3111,3127):
  a=summaries[f'rank8_seed{seed}'];b=summaries[f'rank16_seed{seed}']
  paired.append({'seed':seed,'delta_best_val_ce_rank16_minus8':b['best_validation_ce']-a['best_validation_ce'],
    'relative_throughput_change_rank16_vs8':b['samples_per_second']/a['samples_per_second']-1,
    'delta_peak_gib_rank16_minus8':b['peak_allocated_gib']-a['peak_allocated_gib']})
 return {'status':'pass','kind':'stage31_two_epoch_full512_real_training_4runs_audited',
  'source_git_sha':source,'dataset_source_sha':manifest['source_sha256'],
  'training_exposures_per_model':1024,'unique_train_records':512,'unique_validation_records':96,'heldout_test_records':96,
  'candidates':summaries,'rank_seed_pairs':paired,'chosen_by_validation':selection,
  'frozen_test':{'baseline_assistant_ce':test['base_test_metrics']['assistant_ce_nats'],
    'selected_assistant_ce':test['test_metrics']['assistant_ce_nats'],
    'baseline_ppl':test['base_test_metrics']['assistant_perplexity'],
    'selected_ppl':test['test_metrics']['assistant_perplexity']},
  'limits':['Dolly public and potentially pretrained into base model; no independent data-source transfer',
    'Only two training seeds per rank, insufficient for statistically significant rank benefit',
    'All 512 examples used twice, still a small 8B training budget',
    'Full test used once after validation selection, no semantic correctness claim from CE alone']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 result=analyze(a.root,a.data);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('STAGE31_ALL_FOUR_QLORA_PASS',result['chosen_by_validation']['selected'],'test_ce',round(result['frozen_test']['selected_assistant_ce'],4))
 for name,r in result['candidates'].items():print(name,r['best_epoch'],round(r['best_validation_ce'],4),'tps',round(r['samples_per_second'],3))
if __name__=='__main__':main()
