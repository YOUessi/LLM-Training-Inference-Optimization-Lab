"""External-source IFBench60, independent evaluation of frozen base/LoRA/router.

Never tune or re-rank on IFBench60. Keep raw source/response text locally only:
IFBench test derives from crowdsourced user content and may contain PII.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,random,statistics,time
from pathlib import Path
from llm_opt_lab.training.stage42_risk_router import choose_model,PATTERNS,THRESHOLD
MODELS=('base','lora_sft','request_router')
SRC_SHA='a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85'
MAX_NEW_TOKENS=384

def frozen(private:Path,public:Path):
 manifest=json.loads((public/'stage44_ifbench_manifest.json').read_text())
 content=(private/'stage44_ifbench_external60.jsonl').read_bytes()
 if manifest['source_original_sha256']!=SRC_SHA or hashlib.sha256(content).hexdigest()!=manifest['selected_sha256']:
  raise ValueError('IFBench original source/selection SHA mismatch')
 rows=[json.loads(l) for l in content.decode().splitlines()]
 if len(rows)!=60 or len({str(r['key']) for r in rows})!=60:raise ValueError('IFBench test identities missing')
 key_sha={str(x['key']):x for x in manifest['source_keys_and_prompt_hashes']}
 if len(key_sha)!=60:raise ValueError('manifest identities invalid')
 for r in rows:
  ref=key_sha[str(r['key'])]
  if hashlib.sha256(r['prompt'].encode()).hexdigest()!=ref['prompt_sha256']:
   raise ValueError('IFBench prompt content drift')
  if r['instruction_id_list']!=ref['constraint_ids']:raise ValueError('IFBench constraints drift')
 return rows,manifest

def run_generation(args):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import PeftModel
 rows,m=frozen(args.private,args.data)
 if not (args.adapter/'adapter_model.safetensors').is_file():raise FileNotFoundError('Stage31 frozen selected LoRA missing')
 torch.manual_seed(444044);torch.cuda.manual_seed_all(444044)
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True,
  device_map={'':0},dtype=torch.bfloat16,quantization_config=q,attn_implementation='sdpa')
 tok=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 model=PeftModel.from_pretrained(base,args.adapter,is_trainable=False)
 model.eval()
 output=[];started=time.perf_counter()
 with torch.inference_mode():
  for i,rec in enumerate(rows):
   tokens=tok.apply_chat_template([{'role':'user','content':rec['prompt']}],
     tokenize=True,add_generation_prompt=True,enable_thinking=False)
   if len(tokens)+MAX_NEW_TOKENS>2048:raise ValueError('input cap mismatch source key '+str(rec['key']))
   x=torch.tensor([tokens],device='cuda');att=torch.ones_like(x)
   route=choose_model(rec['prompt'])  # Decision precedes all model outputs and official scoring.
   predictions={};lens={};durations={}
   for model_name in ('base','lora_sft'):
    t0=time.perf_counter()
    if model_name=='base':
     with model.disable_adapter():
      y=model.generate(x,attention_mask=att,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
       pad_token_id=tok.eos_token_id)
    else:
     y=model.generate(x,attention_mask=att,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
       pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    lens[model_name]=int(y.shape[-1]-len(tokens))
    durations[model_name]=time.perf_counter()-t0
    predictions[model_name]=tok.decode(y[0,len(tokens):],skip_special_tokens=True)
   predictions['request_router']=predictions[route.choice]
   lens['request_router']=lens[route.choice]
   durations['request_router']=durations[route.choice]
   output.append({'key':str(rec['key']),'constraint_families':[v.split(':')[0] for v in rec['instruction_id_list']],
     'prompt_sha256':hashlib.sha256(rec['prompt'].encode()).hexdigest(),
     'route':route.choice,'risk_score':route.risk_score,'router_features':list(route.features),
     'responses':predictions,'output_tokens':lens,'generation_seconds':durations,
     'response_sha256':{v:hashlib.sha256(s.encode()).hexdigest() for v,s in predictions.items()}})
   if (i+1)%10==0:print('STAGE44_IFBENCH_REAL_GENERATION',i+1,'/60',flush=True)
 result={'status':'measured','kind':'stage44_external_IFBench60_real_Qwen3_nf4_base_sft_router_generations',
  'selected_IFBench_source_sha256':m['selected_sha256'],
  'original_IFBench_source_sha256':SRC_SHA,'source_IFBench_commit':m['source_commit'],
  'adapter_sha256':hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
  'generated_count':60,'model_comparisons':list(MODELS),
  'configs':{'do_sample':False,'thinking_disabled':True,'max_new_tokens':MAX_NEW_TOKENS,
    'router_threshold':THRESHOLD,'route_uses_only_user_prompt':True},
  'raw_outputs':output,'wall_seconds':time.perf_counter()-started,
  'limitations':['Original IFBench test questions partly from crowd user content; original prompt/generated text only locally private',
   '60 preregistered stratified questions, not official complete 300-source IFBench benchmark',
   'Some long-format tasks may not finish within 384 tokens; cap fraction must be reported',
   'This benchmark is unseen by model-selector, but public data may have been seen by pretrained base',
   'Raw base and LoRA model responses generated sequentially for fair counterfactual audit; router in deployment calls only one']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('STAGE44_60_EXTERNAL_IFBENCH_REAL_GENERATIONS_COMPLETE',len(output),flush=True)


def official_scoring(args):
 from evaluation_lib import InputExample,test_instruction_following_strict,test_instruction_following_loose
 from langdetect import DetectorFactory
 DetectorFactory.seed=0
 rows,m=frozen(args.private,args.data)
 gen=json.loads(args.generated.read_text())
 if gen['status']!='measured' or len(gen['raw_outputs'])!=60 or gen['selected_IFBench_source_sha256']!=m['selected_sha256']:
  raise ValueError('generation did not run on selected original IFBench60')
 output={str(r['key']):r for r in gen['raw_outputs']}
 if len(output)!=60:raise ValueError('duplicate generation IDs')
 # Match the official AllenAI `run_eval.py` call sequence EXACTLY per model:
 # run STRICT over the entire original input list (which strips None kwargs
 # in-place), then run LOOSE over the SAME list in its mutated state.
 # Freshly constructing per-instruction kwargs for loose raises TypeError on
 # original N=None IFBench records, and interleaving all strict/loose per row
 # can change nondeterministic checker behavior relative to official run_eval.
 for rec in rows:
  record=output[str(rec['key'])]
  if hashlib.sha256(rec['prompt'].encode()).hexdigest()!=record['prompt_sha256']:
   raise ValueError('verifier prompt tampering')
  if record['responses']['request_router']!=record['responses'][record['route']]:
   raise ValueError('router was changed post hoc using model answers')
  for name in MODELS:
   text=record['responses'][name]
   if hashlib.sha256(text.encode()).hexdigest()!=record['response_sha256'][name]:
    raise ValueError('generated answer SHA changed')
 model_results={}
 for name in MODELS:
  instances=[InputExample(key=rec['key'],instruction_id_list=list(rec['instruction_id_list']),
      prompt=rec['prompt'],kwargs=[dict(kw) for kw in rec['kwargs']]) for rec in rows]
  # Pass 1: official STRICT mutates each object's kwargs by removing None.
  strict_results=[test_instruction_following_strict(inst,{inst.prompt:output[str(inst.key)]['responses'][name]})
                  for inst in instances]
  # Pass 2: original input objects reused, as done in upstream run_eval.py.
  loose_results=[test_instruction_following_loose(inst,{inst.prompt:output[str(inst.key)]['responses'][name]})
                 for inst in instances]
  model_results[name]=list(zip(strict_results,loose_results))
 per=[]
 for i,rec in enumerate(rows):
  record=output[str(rec['key'])]
  results={}
  for name in MODELS:
   strict,loose=model_results[name][i]
   if len(strict.follow_instruction_list)!=len(rec['instruction_id_list']):
    raise ValueError('official verifier output shape wrong')
   results[name]={'strict':bool(strict.follow_all_instructions),
     'loose':bool(loose.follow_all_instructions),
     'strict_per_instruction':[bool(x) for x in strict.follow_instruction_list],
     'loose_per_instruction':[bool(x) for x in loose.follow_instruction_list],
     'output_tokens':record['output_tokens'][name],
     'hit_cap':bool(record['output_tokens'][name]>=MAX_NEW_TOKENS)}
  per.append({'key':str(rec['key']),'families':list(record['constraint_families']),
    'prompt_sha256':record['prompt_sha256'],'route':record['route'],
    'risk_score':record['risk_score'],'checks':results})
  if (i+1)%15==0:print('STAGE44_IFBENCH_OFFICIAL_VERIFIERS_COMPLETE',i+1,'/60',flush=True)
 denom=sum(len(x['instruction_id_list']) for x in rows)
 aggregate={}
 for name in MODELS:
  aggregate[name]={'strict_correct':sum(x['checks'][name]['strict'] for x in per),
   'strict_prompt_accuracy':statistics.mean(x['checks'][name]['strict'] for x in per),
   'loose_prompt_accuracy':statistics.mean(x['checks'][name]['loose'] for x in per),
   'strict_instruction_accuracy':sum(sum(x['checks'][name]['strict_per_instruction']) for x in per)/denom,
   'hit_output_cap_fraction':statistics.mean(x['checks'][name]['hit_cap'] for x in per),
   'mean_output_tokens':statistics.mean(x['checks'][name]['output_tokens'] for x in per)}
 groups=sorted({item['families'][0] for item in per})
 per_family={f:{name:sum(x['checks'][name]['strict'] for x in per if x['families'][0]==f) for name in MODELS} for f in groups}
 randomizer=random.Random(440044)
 ci={}
 for compare in ('base','lora_sft'):
  group={f:[x for x in per if x['families'][0]==f] for f in groups}
  draws=[]
  for _ in range(3000):
   sample=[arr[randomizer.randrange(len(arr))] for arr in group.values() for _ in range(len(arr))]
   draws.append(statistics.mean(int(x['checks']['request_router']['strict'])-int(x['checks'][compare]['strict']) for x in sample))
  draws.sort();ci[compare]=[draws[75],draws[2924]]
 decision={'strict_at_least_base_minus_2_of60':aggregate['request_router']['strict_correct']>=aggregate['base']['strict_correct']-2,
   'router_strict_greater_than_unscaled_sft':aggregate['request_router']['strict_correct']>aggregate['lora_sft']['strict_correct'],
   'cap_rate_no_more_than_base_plus_0_05':aggregate['request_router']['hit_output_cap_fraction']<=aggregate['base']['hit_output_cap_fraction']+.05}
 result={'status':'measured','kind':'stage44_external_IFBench60_official_strict_loose_OOD',
   'official_IFBench_commit':m['source_commit'],'source_original_sha256':SRC_SHA,
   'selected_input_sha256':m['selected_sha256'],
   'original_generation_sha256':hashlib.sha256(args.generated.read_bytes()).hexdigest(),
   'count':60,'verifier_count':len(per),'model_counts':3,
   'aggregated':aggregate,'first_constraint_family_pass_counts':per_family,
   'matched_3000_stratified_bootstrap_router_minus':ci,
   'preregistered_generalization_acceptance_checks':decision,
   'cross_benchmark_go':all(decision.values()),
   'route_counts':dict(collections.Counter(x['route'] for x in per)),
   'public_anonymized_per_prompt_checks':per,
   'limitations':['External-source IFBench60 is only a fixed 60-case subset of 300, no official full leaderboard score',
    'Official strict/loose verifier checks explicit format, word ratio, sentence etc, not human correctness/production safety',
    'Counterfactual model responses (both generated for each question) are kept locally to avoid republishing potentially private crowd text',
    'No model weights/threshold optimized on IFBench60; reject generalization gate if absent',
    'IFBench public source may still overlap original model pretraining',
    '384 token cap may make some constraints infeasible; cap-hit frequency explicitly measured']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
 print('STAGE44_OFFICIAL_IFBENCH_OUT_OF_DOMAIN_EVALUATION_DONE',aggregate,'GENERALIZATION_ACCEPTANCE',decision,flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--action',choices=('generate','score'),required=True)
 p.add_argument('--private',type=Path,required=True)
 p.add_argument('--data',type=Path,default=Path('data'))
 p.add_argument('--adapter',type=Path,default=Path('evidence/stage31_full512/selected_adapter'))
 p.add_argument('--generated',type=Path)
 p.add_argument('--out',type=Path,required=True)
 args=p.parse_args()
 if args.action=='generate':run_generation(args)
 else:
  if args.generated is None:raise ValueError('--generated required for official score')
  official_scoring(args)
if __name__=='__main__':main()
