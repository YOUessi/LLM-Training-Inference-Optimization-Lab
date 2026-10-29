"""Real frozen 96-record IFEval: base vs two validation-frozen Qwen3 NF4 LoRAs.

The official Google verifier is imported from the exact vendored commit.
This is a controlled *subset* score, never presented as official 541 benchmark.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,random,statistics,time
from pathlib import Path
MODELS=('base','stage26_short','stage31_full')

def sources(data):
 manifest=json.loads((data/'stage36_ifeval_manifest.json').read_text())
 body=(data/'stage36_ifeval_96.jsonl').read_bytes()
 if hashlib.sha256(body).hexdigest()!=manifest['frozen_input_sha256']:raise ValueError('IFEval subset changed')
 records=[json.loads(s) for s in body.decode().splitlines()]
 if len(records)!=96 or len({r['key'] for r in records})!=96:raise ValueError('not 96 frozen inputs')
 return records,manifest

def generation(args):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import PeftModel
 rows,manifest=sources(args.input)
 selection=json.loads((args.stage31/'selection.json').read_text())
 if selection['selected']!='rank16_seed3127' or selection['best_epoch']!=1:
  raise ValueError('Stage31 selection changed after IFEval frozen')
 adapter=args.stage31/(selection['selected']+'_adapter')
 if not adapter.is_dir() or not args.short_adapter.is_dir():raise FileNotFoundError('selected adapter missing')
 for x in (adapter,args.short_adapter):
  if not (x/'adapter_model.safetensors').is_file():raise FileNotFoundError('adapter binary missing')
 torch.manual_seed(360036);torch.cuda.manual_seed_all(360036)
 config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True,device_map={'':0},
  dtype=torch.bfloat16,quantization_config=config,attn_implementation='sdpa')
 tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 model=PeftModel.from_pretrained(base,args.short_adapter,is_trainable=False,adapter_name='stage26')
 model.load_adapter(adapter,adapter_name='stage31',is_trainable=False)
 model.eval()
 outputs=[];start=time.perf_counter()
 with torch.inference_mode():
  for i,row in enumerate(rows):
   encoded=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
   if len(encoded)+manifest['max_generated_tokens']>2048:raise ValueError('context cap violation')
   x=torch.tensor([encoded],device='cuda');mask=torch.ones_like(x)
   generations={};timings={};realized={}
   for mode in MODELS:
    t0=time.perf_counter()
    if mode=='base':
     with model.disable_adapter():
      output=model.generate(x,attention_mask=mask,do_sample=False,max_new_tokens=manifest['max_generated_tokens'],pad_token_id=tokenizer.eos_token_id)
    else:
     model.set_adapter('stage26' if mode=='stage26_short' else 'stage31')
     output=model.generate(x,attention_mask=mask,do_sample=False,max_new_tokens=manifest['max_generated_tokens'],pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    timings[mode]=time.perf_counter()-t0
    realized[mode]=len(output[0])-len(encoded)
    generations[mode]=tokenizer.decode(output[0,len(encoded):],skip_special_tokens=True)
   outputs.append({'key':row['key'],'primary_tier':row['instruction_id_list'][0].split(':')[0],
      'input_sha256':hashlib.sha256(row['prompt'].encode()).hexdigest(),
      'instruction_id_list':row['instruction_id_list'],
      'output_token_counts':realized,'generation_s':timings,
      'responses':generations})
   if (i+1)%12==0:print('STAGE36_GENERATION',i+1,'of',len(rows),flush=True)
 result={'status':'measured','kind':'stage36_ifeval96_model_greedy_responses_three_conditions',
  'source_dataset_sha256':manifest['source_original_sha256'],'frozen_input_sha256':manifest['frozen_input_sha256'],
  'official_ifeval_evaluator_commit':manifest['evaluator_source_commit'],
  'adapters':{'stage26_short':hashlib.sha256((args.short_adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
              'stage31_full':hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()},
  'selected_stage31':selection['selected'],
  'config':{'max_new_tokens':manifest['max_generated_tokens'],'thinking_disabled':True,'do_sample':False,'models':list(MODELS)},
  'cases':outputs,'wall_s':time.perf_counter()-start,
  'limitations':['96 pre-frozen feasible subset of official IFEval 541, not full-dataset leaderboard score',
   'Generated output cap384 can still truncate responses and undercount verifiable instructions',
   'Verifier scores explicit instruction constraints, not factual task correctness or human preference']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('STAGE36_REAL_GENERATIONS_DONE',len(outputs),'x',len(MODELS),'models',flush=True)

def score(args):
 from instruction_following_eval import evaluation_lib
 from langdetect import DetectorFactory
 DetectorFactory.seed=0
 random.seed(360036)
 source,manifest=sources(args.input)
 generated=json.loads(args.generated.read_text())
 if generated['status']!='measured' or generated['frozen_input_sha256']!=manifest['frozen_input_sha256']:
  raise ValueError('generated outputs do not match frozen IFEval prompts')
 if len(generated['cases'])!=len(source)!=96:raise ValueError('missing IFEval prompts')
 answers={x['key']:x for x in generated['cases']}
 if len(answers)!=96:raise ValueError('duplicated IFEval keys')
 per_case=[]
 for row in source:
  x=answers.get(row['key'])
  if x is None or hashlib.sha256(row['prompt'].encode()).hexdigest()!=x['input_sha256']:
   raise ValueError('IFEval input identity mismatch '+str(row['key']))
  if x['instruction_id_list']!=row['instruction_id_list']:raise ValueError('instruction changed')
  instruction=evaluation_lib.InputExample(key=row['key'],instruction_id_list=row['instruction_id_list'],
       prompt=row['prompt'],kwargs=row['kwargs'])
  metrics={}
  for mode in MODELS:
   response=x['responses'][mode]
   if not isinstance(response,str):raise ValueError('nontext response')
   pairs={row['prompt']:response}
   strict=evaluation_lib.test_instruction_following_strict(instruction,pairs)
   loose=evaluation_lib.test_instruction_following_loose(instruction,pairs)
   if len(strict.follow_instruction_list)!=len(row['instruction_id_list']):raise ValueError('official scorer returned wrong shape')
   metrics[mode]={'strict':bool(strict.follow_all_instructions),
    'loose':bool(loose.follow_all_instructions),'strict_each':list(strict.follow_instruction_list),
    'loose_each':list(loose.follow_instruction_list),
    'output_tokens':x['output_token_counts'][mode],
    'hit_cap':x['output_token_counts'][mode]>=manifest['max_generated_tokens']}
  per_case.append({'key':row['key'],'instruction_id_list':row['instruction_id_list'],
    'primary_tier':row['instruction_id_list'][0].split(':')[0],'metrics':metrics})
 overall={}
 for mode in MODELS:
  overall[mode]={
   'strict_prompt_accuracy':sum(x['metrics'][mode]['strict'] for x in per_case)/96,
   'loose_prompt_accuracy':sum(x['metrics'][mode]['loose'] for x in per_case)/96,
   'strict_instruction_accuracy':sum(sum(x['metrics'][mode]['strict_each']) for x in per_case)/sum(len(x['instruction_id_list']) for x in per_case),
   'loose_instruction_accuracy':sum(sum(x['metrics'][mode]['loose_each']) for x in per_case)/sum(len(x['instruction_id_list']) for x in per_case),
   'hit_output_cap_fraction':sum(x['metrics'][mode]['hit_cap'] for x in per_case)/96,
   'mean_realized_output_tokens':statistics.mean(x['metrics'][mode]['output_tokens'] for x in per_case)}
 tiers=sorted({x['primary_tier'] for x in per_case})
 per_tier={k:{m:sum(x['metrics'][m]['strict'] for x in per_case if x['primary_tier']==k)/
                    sum(x['primary_tier']==k for x in per_case) for m in MODELS} for k in tiers}
 rng=random.Random(360036)
 ci={}
 for against in ('base','stage26_short'):
  by_tier={k:[x for x in per_case if x['primary_tier']==k] for k in tiers}
  values=[]
  for _ in range(4000):
   sampled=[rs[rng.randrange(len(rs))] for k,rs in by_tier.items() for _ in range(len(rs))]
   diff=statistics.mean(int(x['metrics']['stage31_full']['strict'])-int(x['metrics'][against]['strict']) for x in sampled)
   values.append(diff)
  values.sort()
  ci[against]={'delta_strict_prompt_accuracy':overall['stage31_full']['strict_prompt_accuracy']-overall[against]['strict_prompt_accuracy'],
    'stratified_4000bootstrap_95pct_ci':[values[100],values[3899]]}
 results={'status':'measured','kind':'stage36_google_official_ifeval96_strict_loose_verification',
  'verified_source_sha256':manifest['frozen_input_sha256'],
  'source_original_sha256':manifest['source_original_sha256'],
  'evaluator_upstream_commit':manifest['evaluator_source_commit'],
  'generator_json_sha256':hashlib.sha256(args.generated.read_bytes()).hexdigest(),
  'counts':{'prompts':96,'instruction_checks':sum(len(x['instruction_id_list']) for x in per_case),'models':3},
  'overall':overall,'by_primary_tier':per_tier,'paired_stratified_uncertainty_full_vs':ci,
  'per_prompt':per_case,
  'limitations':['Only deterministic preregistered feasible subset of 541; cannot report as official IFEval full score',
    'Official Google constraint checkers measure compliance, not semantic factuality',
    'Stratified CIs conditional on a single public prompt sample and single trained checkpoints',
    'Output cap384 creates some impossible long answers; cap-hit explicitly reported',
    'NLTK Punkt Tab official data installed in trusted local dir, no evaluation instructions changed']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(results,indent=2)+'\n')
 print('STAGE36_IFEVAL_OFFICIAL_SCORER_DONE',overall,flush=True)
 print('UNCERTAINTY',ci,flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--action',choices=('generate','score'),required=True)
 p.add_argument('--input',type=Path,default=Path('data'))
 p.add_argument('--stage31',type=Path,required=True)
 p.add_argument('--short-adapter',type=Path,default=Path('evidence/stage26_sft_v2/selected_B_adapter'))
 p.add_argument('--generated',type=Path)
 p.add_argument('--out',type=Path,required=True)
 args=p.parse_args()
 if args.action=='generate':generation(args)
 else:
  if args.generated is None:raise ValueError('--generated required')
  score(args)
if __name__=='__main__':main()
