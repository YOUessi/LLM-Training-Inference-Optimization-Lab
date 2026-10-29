"""Fail-closed SFT analysis: exact frozen splits, selected validation winner, test-once."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path

def analyze(source:Path,data_root:Path):
 names='ABCDE';experiments={n:json.loads((source/f'{n}.json').read_text()) for n in names}
 selection=json.loads((source/'selection.json').read_text());test=json.loads((source/'selected_test.json').read_text())
 manifest=json.loads((data_root/'stage26_manifest.json').read_text())
 if selection['status']!='comparison' or test['status']!='measured' or test['action']!='test':raise ValueError('invalid selection or test status')
 if test['test_metrics']['count']!=96 or test['base_test_metrics']['count']!=96:raise ValueError('incomplete final holdout')
 if test['dataset_sha256']!=manifest['source_sha256']:raise ValueError('dataset source SHA drift')
 subset=['A','B','D','E'];qualified={}
 for name,x in experiments.items():
  if x['status']!='measured' or x['action']!='train' or x['dataset_sha256']!=manifest['source_sha256']:raise ValueError('invalid candidate '+name)
  if x['splits']!={k:v['sha256'] for k,v in manifest['splits'].items()}:raise ValueError('split drift '+name)
  if x['validation_after']['count']!=96 or x['validation_before']['count']!=96:raise ValueError('unpaired validation '+name)
  if x['training']['measured_steps']!=x['config']['steps']-3:raise ValueError('incomplete optimization '+name)
  for fname,sha in x['adapter_sha256'].items():
   if hashlib.sha256((source/f'{name}_adapter'/fname).read_bytes()).hexdigest()!=sha:raise ValueError('adapter SHA mismatch '+name+'/'+fname)
  qualified[name]=x['validation_after']['assistant_ce_nats']<x['validation_before']['assistant_ce_nats']
 if not any(qualified[n] for n in subset):raise ValueError('no validation improvement')
 actual=min((n for n in subset if qualified[n]),key=lambda n:experiments[n]['validation_after']['assistant_ce_nats'])
 if selection['selected']!=actual:raise ValueError('selection disagree with frozen validation')
 if test['config']['rank']!=experiments[actual]['config']['rank'] or test['config']['precision']!=experiments[actual]['config']['precision']:
  raise ValueError('selected test configuration not matched')
 train_metrics={}
 for name,x in experiments.items():
  t=x['training'];before=x['validation_before'];after=x['validation_after']
  train_metrics[name]={'configuration':x['config'],'trainable_parameters':x['trainable_parameters'],
    'base_val_ce':before['assistant_ce_nats'],'val_ce':after['assistant_ce_nats'],
    'val_perplexity':after['assistant_perplexity'],'val_ce_improvement':before['assistant_ce_nats']-after['assistant_ce_nats'],
    'effective_tokens_seen_estimate':None,'throughput_samples_per_second':t['sample_throughput_per_second'],
    'peak_gpu_allocated_gib':t['peak_allocated_gib'],'median_update_ms':t['median_step_ms']}
 return {'status':'pass','source_commit':(source/'source_sha').read_text().strip(),'dataset_sha256':manifest['source_sha256'],
  'valid_train_candidates':list(experiments),'auxiliary_not_used_for_selection':'C',
  'selected':actual,'training':train_metrics,
  'test':{'heldout_records':96,'base_ce':test['base_test_metrics']['assistant_ce_nats'],
          'selected_ce':test['test_metrics']['assistant_ce_nats'],
          'base_perplexity':test['base_test_metrics']['assistant_perplexity'],
          'selected_perplexity':test['test_metrics']['assistant_perplexity'],
          'generated_examples':test['generated_samples']['count'],
          'reference_overlap_f1':test['generated_samples']['mean_reference_token_overlap_f1']},
  'limitations':['Original microbatch2/no-GC was CUDA OOM and is EXCLUDED','Auxiliary C equal approximate examples but twice optimizer updates; not controlled quality comparison',
     'No human quality/semantic judge; reference token overlap is weak','Dataset may overlap pretraining and sample subset is small',
     'Validation 96/test96 both fixed, only test selected once']}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--data-root',type=Path,default=Path('data'));ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 out=analyze(a.source,a.data_root)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')
 print('STAGE26_STRICT_PASS',out['selected'],out['test']);
 for k,v in out['training'].items():print(k,v['configuration'],'val_ce',round(v['val_ce'],4),'tps',round(v['throughput_samples_per_second'],3),'peak_gib',round(v['peak_gpu_allocated_gib'],3))
if __name__=='__main__':main()
