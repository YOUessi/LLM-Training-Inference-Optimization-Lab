"""Guarded inference-time LoRA scaling: preserve instruction following AND SFT CE.

Stage36's already-seen 96 IFEval prompts are excluded. Fit scale on new dev43
using frozen original rules, evaluate ONCE on untouched test101 plus existing
Dolly validation CE. No retraining or IFEval leakage into model weights.
"""
from __future__ import annotations
import argparse,collections,contextlib,hashlib,json,math,random,statistics,time
from pathlib import Path
SCALES=(0.,.25,.5,.75,1.)
DEV_MAX_LOSS_IN_STRICT=2
DEV_MIN_BASE_CE_IMPROVEMENT=.10
DEV_MAX_EXTRA_CAP_HITS=2
MAX_OUTPUT=384
ADAPTER='stage31'

def load_split(data:Path,split:str):
 manifest=json.loads((data/'stage38_manifest.json').read_text())
 details=manifest['splits'][split]
 raw=(data/f'stage38_ifeval_{split}.jsonl').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=details['sha256']:
  raise ValueError('frozen '+split+' SHA mismatch')
 cases=[json.loads(s) for s in raw.decode().splitlines()]
 if len(cases)!=details['count'] or len({r['key'] for r in cases})!=len(cases):
  raise ValueError('stage38 split count/identity error')
 return cases,manifest


def official_scores(rows,generated:dict):
 from instruction_following_eval import evaluation_lib
 from langdetect import DetectorFactory
 DetectorFactory.seed=0
 records=[]
 for r in rows:
  response=generated[r['key']]
  inst=evaluation_lib.InputExample(key=r['key'],instruction_id_list=r['instruction_id_list'],
      prompt=r['prompt'],kwargs=r['kwargs'])
  strict=evaluation_lib.test_instruction_following_strict(inst,{r['prompt']:response})
  loose=evaluation_lib.test_instruction_following_loose(inst,{r['prompt']:response})
  records.append({'key':r['key'],'tier':r['instruction_id_list'][0].split(':')[0],
    'strict':bool(strict.follow_all_instructions),
    'loose':bool(loose.follow_all_instructions),
    'strict_instructions':list(strict.follow_instruction_list),
    'loose_instructions':list(loose.follow_instruction_list)})
 total=sum(len(x['strict_instructions']) for x in records)
 result={'count':len(records),'strict_met_count':sum(x['strict'] for x in records),
  'strict_prompt_fraction':statistics.mean(x['strict'] for x in records),
  'loose_prompt_fraction':statistics.mean(x['loose'] for x in records),
  'strict_instruction_fraction':sum(sum(x['strict_instructions']) for x in records)/total,
  'per_tier':{tier:{'count':sum(x['tier']==tier for x in records),
    'strict_accuracy':statistics.mean(x['strict'] for x in records if x['tier']==tier)}
    for tier in sorted({x['tier'] for x in records})},'per_case':records}
 return result


def set_scale_factor(model,alpha:float,adapter_name=ADAPTER):
 """Temporarily scale all LoRA residuals, restore exactly on exit."""
 from peft.tuners.lora.layer import LoraLayer
 if not 0<alpha<=1:raise ValueError('positive LoRA residual alpha required')
 selected=[]
 for module in model.modules():
  if isinstance(module,LoraLayer) and adapter_name in module.scaling:
   previous=module.scaling[adapter_name]
   if not isinstance(previous,(int,float)) or not math.isfinite(previous):raise ValueError('invalid original LoRA scaling')
   selected.append((module,float(previous)))
 if len(selected)!=72:raise ValueError(f'Qwen3 q/v 36-layer expected 72 LoRA modules; found {len(selected)}')
 @contextlib.contextmanager
 def temporary():
  try:
   for module,initial in selected:module.scaling[adapter_name]=initial*alpha
   yield
  finally:
   for module,initial in selected:module.scaling[adapter_name]=initial
 return temporary()


def load_model(adapter:Path):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import PeftModel
 if not adapter.is_dir() or not (adapter/'adapter_model.safetensors').exists():
  raise FileNotFoundError('missing frozen adapter')
 torch.manual_seed(38038);torch.cuda.manual_seed_all(38038)
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True,
   device_map={'':0},dtype=torch.bfloat16,quantization_config=quant,attn_implementation='sdpa')
 base.config.use_cache=False
 tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 model=PeftModel.from_pretrained(base,adapter,adapter_name=ADAPTER,is_trainable=False)
 model.eval()
 return model,tokenizer

@contextlib.contextmanager
def scaled(model,alpha):
 if alpha==0:
  with model.disable_adapter():yield
 else:
  with set_scale_factor(model,alpha):yield


def generate_all(model,tokenizer,rows,alpha,source_label):
 import torch
 results=[]
 model.eval()
 with scaled(model,alpha),torch.inference_mode():
  for i,row in enumerate(rows):
   encoded=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],
       tokenize=True,add_generation_prompt=True,enable_thinking=False)
   if len(encoded)+MAX_OUTPUT>2048:raise ValueError('prompt+generation exceeds tested context limit')
   ids=torch.tensor([encoded],device='cuda');mask=torch.ones_like(ids)
   generated=model.generate(ids,attention_mask=mask,do_sample=False,max_new_tokens=MAX_OUTPUT,
     pad_token_id=tokenizer.eos_token_id)
   prediction=tokenizer.decode(generated[0,len(encoded):],skip_special_tokens=True)
   results.append({'key':row['key'],'tier':row['instruction_id_list'][0].split(':')[0],
     'input_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
     'response':prediction,'response_sha256':hashlib.sha256(prediction.encode()).hexdigest(),
     'generated_tokens':int(generated.shape[-1]-len(encoded)),
     'at_cap':bool(generated.shape[-1]-len(encoded)>=MAX_OUTPUT)})
   if (i+1)%10==0:print('STAGE38_GENERATED',source_label,'scale',alpha,i+1,'/',len(rows),flush=True)
 return results


def answer_ce(model,tokenizer,data:Path,alpha):
 import torch
 from llm_opt_lab.training.stage26_real_sft import collate,load_split as tokenize_dolly
 manifest=json.loads((data/'stage26_manifest.json').read_text())
 file=data/'stage26_validation.jsonl'
 if hashlib.sha256(file.read_bytes()).hexdigest()!=manifest['splits']['validation']['sha256']:
  raise ValueError('Dolly validation SHA changed')
 rows=tokenize_dolly(file,tokenizer)
 if len(rows)!=96:raise ValueError('Dolly validation incomplete')
 total=0.;ntok=0
 with scaled(model,alpha),torch.inference_mode():
  for r in rows:
   batch=collate([r],tokenizer.eos_token_id)
   loss=float(model(**batch,use_cache=False).loss)
   if not math.isfinite(loss):raise ValueError('Dolly crossentropy invalid')
   total+=loss*r['answer_count'];ntok+=r['answer_count']
 return {'count':len(rows),'target_tokens':ntok,'ce_nats':total/ntok,
   'perplexity':math.exp(total/ntok)}

def fold_scores(rows,outputs):
 bykey={x['key']:x['response'] for x in outputs}
 if len(bykey)!=len(rows):raise ValueError('one response required for each source key')
 official=official_scores(rows,bykey)
 cap=sum(x['at_cap'] for x in outputs)
 official.update({'hit_cap_count':cap,'hit_cap_fraction':cap/len(rows),
                  'mean_realized_output_tokens':statistics.mean(x['generated_tokens'] for x in outputs)})
 return official


def select_dev(out:Path,data:Path,adapter:Path):
 rows,manifest=load_split(data,'dev')
 if len(rows)!=43:raise ValueError('dev must remain 43')
 if out.exists():raise FileExistsError('dev output already exists; no overwrite')
 out.mkdir(parents=True)
 source_git=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip()
 (out/'source_sha').write_text(source_git+'\n')
 model,tokenizer=load_model(adapter)
 metrics={}
 for alpha in SCALES:
  key=f'{alpha:.2f}'
  outputs=generate_all(model,tokenizer,rows,alpha,'dev')
  folder=out/f'dev_alpha_{key}.json'
  folder.write_text(json.dumps({'alpha':alpha,'source_input_sha256':manifest['splits']['dev']['sha256'],
       'responses':outputs},indent=2,ensure_ascii=False)+'\n')
  metrics[key]={'alpha':alpha,'ifeval_dev':fold_scores(rows,outputs),
       'dolly_val':answer_ce(model,tokenizer,data,alpha)}
  print('STAGE38_DEV',key,'strict_met',metrics[key]['ifeval_dev']['strict_met_count'],
    'DOLLY_VAL_CE',round(metrics[key]['dolly_val']['ce_nats'],4),
    'hit_cap',metrics[key]['ifeval_dev']['hit_cap_count'],flush=True)
 baseline=metrics['0.00']
 feasible=[]
 for key,result in metrics.items():
  if result['alpha']==0:continue
  constraints={
   'stricter_than_base_allow_two_misses':result['ifeval_dev']['strict_met_count']>=baseline['ifeval_dev']['strict_met_count']-DEV_MAX_LOSS_IN_STRICT,
   'at_most_two_additional_output_cap_hits':result['ifeval_dev']['hit_cap_count']<=baseline['ifeval_dev']['hit_cap_count']+DEV_MAX_EXTRA_CAP_HITS,
   'dolly_ce_gain_at_least_0_10_nats':result['dolly_val']['ce_nats']<=baseline['dolly_val']['ce_nats']-DEV_MIN_BASE_CE_IMPROVEMENT}
  result['guardrail_checks']=constraints
  if all(constraints.values()):feasible.append(key)
 selected=min(feasible,key=lambda key:(metrics[key]['dolly_val']['ce_nats'],metrics[key]['alpha'])) if feasible else '0.00'
 decision={'status':'dev_approved_for_holdout_not_production','kind':'stage38_frozen_ifeval_dev_selection',
  'source_sha':source_git,'previously_seen_stage36_keys_excluded':96,
  'new_dev_count':43,'holdout_count':101,
  'source_dataset_sha256':manifest['original_source_sha256'],
  'split_sha256':{split:x['sha256'] for split,x in manifest['splits'].items()},
  'adapter_sha256':hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
  'scale_grid':list(SCALES),'metrics':metrics,
  'predeclared_guardrails':{'max_ifeval_dev_lost_strict_prompts':DEV_MAX_LOSS_IN_STRICT,
   'max_additional_output_cap_hits':DEV_MAX_EXTRA_CAP_HITS,
   'minimum_dolly_validation_ce_gain_nats':DEV_MIN_BASE_CE_IMPROVEMENT,
   'selection':'Among nonzero feasible strengths minimize Dolly validation CE; no feasible strength -> base0',
   'Dolly_source':'stage26_validation.jsonl 96 fixed before tests'},
  'feasible_scales':feasible,'selected_key':selected,
  'selected_alpha':metrics[selected]['alpha'],
  'guardrail_passed_with_adaptation':bool(feasible),
  'holdout_not_loaded_or_used_for_selection':True,
  'limitations':['Dev43 from official IFEval already public but identities disjoint from Stage36 and final101',
      'All scores source-adapted to fixed group quotas, scale selection not a broadly trained algorithm',
      'LoRA residual scale manipulation is an inference-time configuration, not a trained or validated new architecture']}
 (out/'frozen_dev_selection.json').write_text(json.dumps(decision,indent=2)+'\n')
 print('STAGE38_FROZEN_DEV_DECISION',selected,'feasible_adapted',feasible,flush=True)


def resample_paired(rows:dict,against:str,rng:random.Random):
 alltiers=sorted({row['tier'] for row in rows.values()})
 tiers={tier:[r for r in rows.values() if r['tier']==tier] for tier in alltiers}
 vals=[]
 for _ in range(3000):
  drawn=[group[rng.randrange(len(group))] for group in tiers.values() for _ in range(len(group))]
  vals.append(statistics.mean(int(row['strict']['selected'])-int(row['strict'][against]) for row in drawn))
 vals.sort();return [vals[75],vals[2924]]


def test_locked(out:Path,data:Path,adapter:Path):
 source,manifest=load_split(data,'test')
 if len(source)!=101:raise ValueError('test must remain 101')
 decision=json.loads((out/'frozen_dev_selection.json').read_text())
 if decision['status']!='dev_approved_for_holdout_not_production' or not decision['holdout_not_loaded_or_used_for_selection']:
  raise ValueError('no dev-frozen scale selection')
 if decision['split_sha256']['test']!=manifest['splits']['test']['sha256']:
  raise ValueError('holdout changed after dev scale selection')
 if decision['adapter_sha256']!=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest():
  raise ValueError('adapter changed after selection')
 if (out/'final_untouched101_test.json').exists():raise FileExistsError('final test already exists')
 # Test ALL three preregistered controls, without selecting alpha from test responses.
 test_alphas=list(dict.fromkeys([0.,1.,float(decision['selected_alpha'])]))
 model,tokenizer=load_model(adapter)
 observed={};reports={}
 for alpha in test_alphas:
  key=f'{alpha:.2f}'
  observed[key]=generate_all(model,tokenizer,source,alpha,'untouched101')
  reports[key]=fold_scores(source,observed[key])
  print('STAGE38_TEST_ARM',key,'strict',reports[key]['strict_met_count'],'/',len(source),
        'cap',reports[key]['hit_cap_count'],flush=True)
 selectkey=decision['selected_key']
 records={}
 for row in source:
  key=row['key'];tier=row['instruction_id_list'][0].split(':')[0]
  records[key]={'key':key,'tier':tier,'strict':{
    alias:next(x for x in reports[alpha]['per_case'] if x['key']==key)['strict']
    for alias,alpha in [('base','0.00'),('full','1.00'),('selected',selectkey)]}}
 rng=random.Random(380038)
 cis={name:resample_paired(records,name,rng) for name in ('base','full')}
 result={'status':'measured','kind':'stage38_unseen101_rule_following_scale_guardrail_test',
  'dev_decision_sha256':hashlib.sha256((out/'frozen_dev_selection.json').read_bytes()).hexdigest(),
  'test_input_sha256':manifest['splits']['test']['sha256'],'new_holdout_count':101,
  'previously_seen_stage36_excluded':96,'dev43_disjoint':True,
  'selected_alpha':float(decision['selected_alpha']),'feasible_adapted_on_dev':decision['guardrail_passed_with_adaptation'],
  'tested_alpha_keys':list(reports),'scores':reports,'paired_test_CIs_selected_minus':cis,
  'selected_and_comparators':{'base_strict':reports['0.00']['strict_prompt_fraction'],
   'full_unscaled_strict':reports['1.00']['strict_prompt_fraction'],
   'selected_strict':reports[selectkey]['strict_prompt_fraction']},
  'source_adapter_git_sha256':decision['adapter_sha256'],
  'frozen_no_test_selection':True,
  'limitations':['One original trained adapter, five dev-scanned LoRA residual multipliers; no model fine-tuning',
    'Test101 official IFEval source and distinct from previously observed Stage36 96 and dev43',
    'Original model may have seen public IFEval before pretraining; not human correctness/safety',
    'Bootstrap over fixed test question identities not independent training seeds',
    'Unscaled full adapter and original base are predeclared controls; final test was used only once']}
 (out/'final_untouched101_test.json').write_text(json.dumps(result,indent=2)+'\n')
 # Privacy not involved: IFEval is Apache, raw prompt/output can be released for audit.
 (out/'final_101_outputs.json').write_text(json.dumps({'status':'measured','test_input_sha256':manifest['splits']['test']['sha256'],
  'generation_by_scale':observed},indent=2,ensure_ascii=False)+'\n')
 print('STAGE38_UNTOUCHED101_TEST_COMPLETE',result['selected_and_comparators'],cis,flush=True)


def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--action',choices=('dev','test'),required=True)
 ap.add_argument('--data',type=Path,default=Path('data'))
 ap.add_argument('--adapter',type=Path,default=Path('evidence/stage31_full512/selected_adapter'))
 ap.add_argument('--out',type=Path,required=True)
 args=ap.parse_args()
 if args.action=='dev':select_dev(args.out,args.data,args.adapter)
 else:test_locked(args.out,args.data,args.adapter)
if __name__=='__main__':main()
