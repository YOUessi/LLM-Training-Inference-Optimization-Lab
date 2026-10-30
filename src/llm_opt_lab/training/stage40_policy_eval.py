"""Guarded selection of format-rehearsal QLoRA vs base on IFEval dev25 and Dolly val96.

The final60 source file is read ONLY after a selection JSON is frozen. Prior
observed Stage36 and Stage38 IFEval identities are excluded from both.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,random,statistics,time
from pathlib import Path
from llm_opt_lab.training.stage38_adapter_strength import official_scores
from llm_opt_lab.training.stage26_real_sft import MODEL
SEEDS=(4011,4027)
MIXES=(0.,.25,.50)
GEN_TOKENS=384
MAX_DEV_LOSS_OF_STRICT=1
MAX_EXTRA_HITCAP=2
MIN_GAIN_DOLLY_CE=.10

def name(seed,mix):return f'mix{int(mix*100):02d}_seed{seed}'

def corpus(folder:Path,split:str):
 manifest=json.loads((folder/'stage40_manifest.json').read_text())
 data=manifest['ifeval_dev_test'][split]
 content=(folder/f'stage40_ifeval_{split}.jsonl').read_bytes()
 if hashlib.sha256(content).hexdigest()!=data['sha256']:raise ValueError('stage40 split sha changed')
 rows=[json.loads(s) for s in content.decode().splitlines()]
 if len(rows)!=data['records'] or len({r['key'] for r in rows})!=len(rows):raise ValueError('stage40 identity/count failure')
 return rows,manifest

def setup(adapter_root:Path):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from peft import PeftModel
 meta=json.loads((adapter_root/'selection.json').read_text()) if (adapter_root/'selection.json').exists() else None
 selected_original=Path('evidence/stage31_full512/selected_adapter')
 torch.manual_seed(4040);torch.cuda.manual_seed_all(4040)
 config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,device_map={'':0},
  dtype=torch.bfloat16,quantization_config=config,attn_implementation='sdpa')
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 model=PeftModel.from_pretrained(base,selected_original,is_trainable=False,adapter_name='old_full')
 specs={}
 for seed in SEEDS:
  for mix in MIXES:
   label=name(seed,mix)
   source=adapter_root/(label+'.json')
   data=json.loads(source.read_text())
   if data['status']!='measured' or data['config']['seed']!=seed or data['config']['mix']!=mix:
    raise ValueError('not a verified trained replay candidate '+label)
   model.load_adapter(adapter_root/(label+'_adapter'),adapter_name=label,is_trainable=False)
   specs[label]=data
 model.eval()
 return model,tokenizer,specs

def make_answers(model,tokenizer,rows,adapter,stage):
 import torch
 result=[]
 with torch.inference_mode():
  if adapter=='base':
   from contextlib import nullcontext
   context=model.disable_adapter()
  else:
   model.set_adapter(adapter)
   from contextlib import nullcontext
   context=nullcontext()
  with context:
   for i,record in enumerate(rows):
    ids=tokenizer.apply_chat_template([{'role':'user','content':record['prompt']}],tokenize=True,
      add_generation_prompt=True,enable_thinking=False)
    if len(ids)+GEN_TOKENS>2048:raise ValueError('unapproved long IFEval request')
    x=torch.tensor([ids],device='cuda');mask=torch.ones_like(x)
    y=model.generate(x,attention_mask=mask,max_new_tokens=GEN_TOKENS,do_sample=False,
       pad_token_id=tokenizer.eos_token_id)
    text=tokenizer.decode(y[0,len(ids):],skip_special_tokens=True)
    result.append({'key':record['key'],'family':record['instruction_id_list'][0].split(':')[0],
     'prompt_sha256':hashlib.sha256(record['prompt'].encode()).hexdigest(),
     'answer':text,'answer_sha256':hashlib.sha256(text.encode()).hexdigest(),
     'generated_tokens':len(y[0])-len(ids),'hit_cap':len(y[0])-len(ids)>=GEN_TOKENS})
    if (i+1)%10==0:print('STAGE40_GENERATED',stage,adapter,i+1,'/',len(rows),flush=True)
 return result

def evaluate(rows,outputs):
 response={x['key']:x['answer'] for x in outputs}
 if len(response)!=len(rows):raise ValueError('unpaired responses')
 data=official_scores(rows,response)
 data.update({'hit_cap_count':sum(r['hit_cap'] for r in outputs),'mean_answer_tokens':statistics.mean(r['generated_tokens'] for r in outputs),
  'hit_cap_fraction':sum(r['hit_cap'] for r in outputs)/len(outputs)})
 return data

def dev(args):
 rows,manifest=corpus(args.data,'dev')
 if len(rows)!=25:raise ValueError('dev must be 25')
 if (args.root/'dev_selection.json').exists():raise FileExistsError('dev selection exists')
 model,tokenizer,specs=setup(args.root)
 arms=['base','old_full']+list(specs)
 output={};metrics={}
 for label in arms:
  result=make_answers(model,tokenizer,rows,label,'dev')
  output[label]=result;quality=evaluate(rows,result)
  ce=(specs[label]['best_validation_ce'] if label in specs else
        specs[name(4011,0.)]['baseline_validation']['assistant_ce_nats'] if label=='base' else
        json.loads(Path('evidence/stage31_full512/analysis.json').read_text())['candidates']['rank16_seed3127']['best_validation_ce'])
  metrics[label]={'strict':quality['strict_met_count'],'strict_fraction':quality['strict_prompt_fraction'],
   'loose_fraction':quality['loose_prompt_fraction'],'capped':quality['hit_cap_count'],'Dolly_validation_ce':ce,
   'per_family':quality['per_tier']}
  print('STAGE40_DEV',label,metrics[label],flush=True)
 base=metrics['base'];feasible=[]
 for label in specs:
  row=metrics[label]
  checks={'strict_at_least_base_minus_one':row['strict']>=base['strict']-MAX_DEV_LOSS_OF_STRICT,
    'cap_at_most_base_plus_two':row['capped']<=base['capped']+MAX_EXTRA_HITCAP,
    'Dolly_validation_ce_improves_by_at_least_point10':row['Dolly_validation_ce']<=base['Dolly_validation_ce']-MIN_GAIN_DOLLY_CE}
  metrics[label]['guardrails']=checks
  if all(checks.values()):feasible.append(label)
 selection=min(feasible,key=lambda name:(metrics[name]['Dolly_validation_ce'],name)) if feasible else 'base'
 # Never optimize with a single Stage40 untouched test response.
 decision={'status':'dev_frozen_no_final_test_access','source_data_sha256':manifest['dolly_format_replay_sha256'],
  'dev25_sha256':manifest['ifeval_dev_test']['dev']['sha256'],
  'untouched60_sha256':manifest['ifeval_dev_test']['test']['sha256'],
  'all_stage36_and_stage38_ifeval_keys_excluded_count':240,
  'fixed_candidate_arms':arms,'training_candidates':list(specs),
  'metrics':metrics,'guardrails':{'max_dev_strict_lost_vs_base':MAX_DEV_LOSS_OF_STRICT,
    'max_extra_cap384_hits_vs_base':MAX_EXTRA_HITCAP,'Dolly_val_CE_improvement_at_least':MIN_GAIN_DOLLY_CE,
    'tie_break':'minimum Dolly CE among feasible format/control QLoRA models; none feasible -> base'},
  'feasible_adapted':feasible,'selected':selection,
  'selected_adapted':selection!='base','final_test_unaccessed':True}
 (args.root/'dev_selection.json').write_text(json.dumps(decision,indent=2)+'\n')
 (args.root/'dev_outputs.json').write_text(json.dumps({'status':'measured','dev_sha256':decision['dev25_sha256'],
    'answers':output},indent=2,ensure_ascii=False)+'\n')
 print('STAGE40_DEV_SELECTION_FROZEN',selection,'admissible',feasible,flush=True)

def test(args):
 rows,manifest=corpus(args.data,'test')
 if len(rows)!=60:raise ValueError('test must remain 60')
 decision=json.loads((args.root/'dev_selection.json').read_text())
 if decision['status']!='dev_frozen_no_final_test_access' or not decision['final_test_unaccessed']:
  raise ValueError('not a legitimate dev-frozen selection')
 if decision['untouched60_sha256']!=manifest['ifeval_dev_test']['test']['sha256']:raise ValueError('test changed after dev selection')
 if (args.root/'final_test60_official.json').exists():raise FileExistsError('final test already consumed')
 model,tokenizer,specs=setup(args.root)
 labels=list(dict.fromkeys(['base','old_full',decision['selected']]))
 results={};stats={}
 for label in labels:
  out=make_answers(model,tokenizer,rows,label,'final_unseen60')
  results[label]=out;stats[label]=evaluate(rows,out)
  print('STAGE40_TEST',label,stats[label]['strict_met_count'],'/',len(rows),'CAP',stats[label]['hit_cap_count'],flush=True)
 bykey={r['key']:r for r in rows}
 scores={label:{z['key']:next(x['strict'] for x in stats[label]['per_case'] if x['key']==z['key']) for z in rows} for label in labels}
 rng=random.Random(404040)
 intervals={}
 tiers={family:[r['key'] for r in rows if r['instruction_id_list'][0].split(':')[0]==family]
   for family in sorted({r['instruction_id_list'][0].split(':')[0] for r in rows})}
 for compare in ('base','old_full'):
  draws=[]
  for _ in range(3000):
   keys=[ids[rng.randrange(len(ids))] for ids in tiers.values() for _ in range(len(ids))]
   draws.append(statistics.mean(int(scores[decision['selected']][k])-int(scores[compare][k]) for k in keys))
  draws.sort()
  intervals[compare]=[draws[75],draws[2924]]
 report={'status':'measured','kind':'stage40_format_replay_unseen60_ifeval_test',
  'dev_selection_sha256':hashlib.sha256((args.root/'dev_selection.json').read_bytes()).hexdigest(),
  'test60_sha256':manifest['ifeval_dev_test']['test']['sha256'],'dev25_disjoint':True,
  'prior240_ifeval_keys_excluded':True,'selected':decision['selected'],
  'selected_adapted':decision['selected_adapted'],
  'tested_labels':labels,'full_official_scores':stats,
  'paired_stratified_CIs_selected_minus':intervals,
  'limitations':['Only 60 untouched IFEval examples from five instruction categories, 240 observed previous keys excluded',
   'SFT constraint targets are synthetically transformed Dolly answers, not human annotated',
   'Official constraint adherence on source subset is not factual/safety quality',
   'One Qwen3-8B base and two training seeds; conditional bootstraps not population-level significance',
   'No candidate was selected based on the final60; original base fallback enforced if dev fails']}
 (args.root/'final_test60_official.json').write_text(json.dumps(report,indent=2)+'\n')
 (args.root/'final_outputs60.json').write_text(json.dumps({'status':'measured','test_sha256':report['test60_sha256'],
  'answers':results},indent=2,ensure_ascii=False)+'\n')
 print('STAGE40_UNTOUCHED_60_COMPLETE',decision['selected'],{k:v['strict_prompt_fraction'] for k,v in stats.items()},flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--action',choices=('dev','test'),required=True)
 p.add_argument('--data',type=Path,default=Path('data'))
 p.add_argument('--root',type=Path,required=True)
 a=p.parse_args()
 if a.action=='dev':dev(a)
 else:test(a)
if __name__=='__main__':main()
