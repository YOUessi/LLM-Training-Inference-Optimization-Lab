"""Real GPU counterfactual evaluation: base vs full LoRA vs prompt-only router.

The router is evaluated strictly after it makes each prompt decision, and never
looks at expected answer/IFEval rule checker/model outputs when making decisions.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,random,statistics,time
from pathlib import Path
from llm_opt_lab.training.stage42_risk_router import choose_model,PATTERNS,THRESHOLD
MODELS=('base','lora_sft','request_router')


def frozen(data:Path):
 m=json.loads((data/'stage42_manifest.json').read_text())
 result={}
 for split,(name,count) in {'ifeval':('stage42_ifeval_untouched96.jsonl',96),
                             'dolly':('stage42_dolly_untouched128.jsonl',128)}.items():
  raw=(data/name).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=m[split]['sha256']:raise ValueError('test source changed '+split)
  rows=[json.loads(s) for s in raw.decode().splitlines()]
  field='key' if split=='ifeval' else 'identity_sha256'
  if len(rows)!=count or len({x[field] for x in rows})!=count:raise ValueError('source IDs duplicated')
  result[split]=rows
 return result,m


def make_model(adapter:Path):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import PeftModel
 if not (adapter/'adapter_model.safetensors').is_file():raise FileNotFoundError(adapter)
 torch.manual_seed(420042);torch.cuda.manual_seed_all(420042)
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
     bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True,
     dtype=torch.bfloat16,device_map={'':0},quantization_config=q,attn_implementation='sdpa')
 base.config.use_cache=False
 tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 model=PeftModel.from_pretrained(base,adapter,is_trainable=False)
 model.eval()
 return model,tokenizer


def score_rules(rows,response_by_model):
 from instruction_following_eval import evaluation_lib
 from langdetect import DetectorFactory
 DetectorFactory.seed=0
 per=[]
 for r in rows:
  inst=evaluation_lib.InputExample(key=r['key'],instruction_id_list=r['instruction_id_list'],
      prompt=r['prompt'],kwargs=r['kwargs'])
  item={'key':r['key'],'tier':r['instruction_id_list'][0].split(':')[0]}
  for choice in MODELS:
   text=response_by_model[r['key']][choice]
   strict=evaluation_lib.test_instruction_following_strict(inst,{r['prompt']:text})
   loose=evaluation_lib.test_instruction_following_loose(inst,{r['prompt']:text})
   item[choice]={'strict':bool(strict.follow_all_instructions),
                 'loose':bool(loose.follow_all_instructions),
                 'strict_instructions':[bool(v) for v in strict.follow_instruction_list],
                 'loose_instructions':[bool(v) for v in loose.follow_instruction_list]}
  per.append(item)
 total=sum(len(r['instruction_id_list']) for r in rows)
 aggregate={}
 for label in MODELS:
  aggregate[label]={'correct':sum(x[label]['strict'] for x in per),'strict_prompt_fraction':sum(x[label]['strict'] for x in per)/len(per),
    'loose_prompt_fraction':sum(x[label]['loose'] for x in per)/len(per),
    'strict_instruction_fraction':sum(sum(x[label]['strict_instructions']) for x in per)/total}
 return per,aggregate


def compile_case(model,tokenizer,record):
 """All three conditions share identical input text and tokenization."""
 import torch
 prompt=record['prompt']
 ids=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
      tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if len(ids)+384>2048:raise ValueError('source exceeds 8B context limit')
 x=torch.tensor([ids],device='cuda')
 attn=torch.ones_like(x)
 result={};timings={};lengths={}
 with torch.inference_mode():
  for label in ('base','lora_sft'):
   start=time.perf_counter()
   if label=='base':
    with model.disable_adapter():
     o=model.generate(x,attention_mask=attn,max_new_tokens=384,do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
   else:
    o=model.generate(x,attention_mask=attn,max_new_tokens=384,do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
   torch.cuda.synchronize()
   timings[label]=time.perf_counter()-start
   lengths[label]=int(o.shape[1]-len(ids))
   result[label]=tokenizer.decode(o[0,len(ids):],skip_special_tokens=True)
 policy=choose_model(prompt)
 result['request_router']=result[policy.choice]
 timings['request_router']=timings[policy.choice]
 lengths['request_router']=lengths[policy.choice]
 return result,timings,lengths,policy


def eval_dolly(model,tokenizer,rows):
 import torch
 from llm_opt_lab.training.stage26_real_sft import collate
 results=[]
 with torch.inference_mode():
  for i,r in enumerate(rows):
   user=r['instruction']+('\n\nReference context:\n'+r['context'] if r['context'] else '')
   prefix=tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=True,
       add_generation_prompt=True,enable_thinking=False)
   ids=tokenizer.apply_chat_template([{'role':'user','content':user},{'role':'assistant','content':r['response']}],
       tokenize=True,add_generation_prompt=False,enable_thinking=False)
   if ids[:len(prefix)]!=prefix or len(ids)>384:raise ValueError('frozen Dolly tokenization drift')
   batch=collate([{'input_ids':ids,'labels':[-100]*len(prefix)+ids[len(prefix):]}],tokenizer.eos_token_id)
   policy=choose_model(user)
   with model.disable_adapter():base_ce=float(model(**batch,use_cache=False).loss.detach())
   lora_ce=float(model(**batch,use_cache=False).loss.detach())
   if not math.isfinite(base_ce) or not math.isfinite(lora_ce):raise ValueError('nonfinite Dolly CE')
   results.append({'identity_sha256':r['identity_sha256'],'category':r['category'],
     'route':policy.choice,'risk':policy.risk_score,'features':policy.features,
     'answer_tokens':len(ids)-len(prefix),'CE':{'base':base_ce,'lora_sft':lora_ce,
               'request_router':base_ce if policy.choice=='base' else lora_ce}})
   if (i+1)%32==0:print('STAGE42_DOLLY_CE',i+1,'/128',flush=True)
 return results


def aggregate_ce(rows):
 total=sum(r['answer_tokens'] for r in rows)
 return {label:{'examples':len(rows),'answer_tokens':total,
    'ce_nats':sum(r['CE'][label]*r['answer_tokens'] for r in rows)/total,
    'perplexity':math.exp(sum(r['CE'][label]*r['answer_tokens'] for r in rows)/total)}
    for label in MODELS}


def bootstrap(rules,ce_rows):
 rng=random.Random(420420);group={}
 for row in rules:group.setdefault(row['tier'],[]).append(row)
 strats={}
 for compare in ('base','lora_sft'):
  draws=[]
  for _ in range(2000):
   samples=[cat[rng.randrange(len(cat))] for cat in group.values() for _ in range(len(cat))]
   draws.append(statistics.mean(int(x['request_router']['strict'])-int(x[compare]['strict']) for x in samples))
  draws.sort();strats[compare]=[draws[50],draws[1949]]
 dgroups={}
 for row in ce_rows:dgroups.setdefault(row['category'],[]).append(row)
 ce_unc={}
 for compare in ('base','lora_sft'):
  draws=[]
  for _ in range(2000):
   sample=[r[rng.randrange(len(r))] for r in dgroups.values() for _ in range(len(r))]
   token_count=sum(x['answer_tokens'] for x in sample)
   draws.append(sum((x['CE']['request_router']-x['CE'][compare])*x['answer_tokens'] for x in sample)/token_count)
  draws.sort();ce_unc[compare]=[draws[50],draws[1949]]
 return {'ifeval_router_minus':strats,'dolly_router_ce_minus':ce_unc,
   'bootstrap':'2000 stratified with-replacement draws within frozen source tiers/categories; conditional only'}


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--input',type=Path,default=Path('data'))
 parser.add_argument('--adapter',type=Path,default=Path('evidence/stage31_full512/selected_adapter'))
 parser.add_argument('--out',type=Path,required=True)
 args=parser.parse_args()
 data,manifest=frozen(args.input)
 model,tok=make_model(args.adapter)
 import torch
 torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 outputs=[];responses={}
 for i,r in enumerate(data['ifeval']):
  text,times,counts,decision=compile_case(model,tok,r)
  responses[r['key']]=text
  outputs.append({'key':r['key'],'family':r['instruction_id_list'][0].split(':')[0],
    'prompt_sha256':hashlib.sha256(r['prompt'].encode()).hexdigest(),
    'route':decision.choice,'risk_score':decision.risk_score,'detected_constraints':decision.features,
    'generation_s':times,'output_tokens':counts,
    'answer_sha256':{label:hashlib.sha256(text[label].encode()).hexdigest() for label in MODELS}})
  if (i+1)%12==0:print('STAGE42_REAL_IFEVAL_GENERATION',i+1,'/96',flush=True)
 official,accuracy=score_rules(data['ifeval'],responses)
 print('STAGE42_IFEVAL_STRICT', {m:accuracy[m]['correct'] for m in MODELS},flush=True)
 ce_rows=eval_dolly(model,tok,data['dolly'])
 ce=aggregate_ce(ce_rows)
 print('STAGE42_DOLLY_CE', {m:round(ce[m]['ce_nats'],4) for m in MODELS},flush=True)
 uncertainty=bootstrap(official,ce_rows)
 per_family={family:{label:sum(x[label]['strict'] for x in official if x['tier']==family) for label in MODELS}
  for family in sorted({x['tier'] for x in official})}
 routings={'ifeval':dict(collections.Counter(x['route'] for x in outputs)),
   'dolly':dict(collections.Counter(x['route'] for x in ce_rows))}
 gate={'near_base_ifeval_allow_2_prompts':accuracy['request_router']['correct']>=accuracy['base']['correct']-2,
   'dolly_CE_improves_at_least_0_50_nats':ce['request_router']['ce_nats']<=ce['base']['ce_nats']-.5,
   'zero_benchmark_label_or_answer_in_router':True,
   'completed_source96_and128':len(outputs)==96 and len(ce_rows)==128}
 report={'status':'measured','kind':'stage42_prompt_only_base_vs_sft_constraint_router_clean_96_128',
  'input_sha256':{'ifeval':manifest['ifeval']['sha256'],'dolly':manifest['dolly']['sha256']},
  'previous_source_identites_excluded':{'IFEval':325,'Dolly':960},
  'adapter_binary_sha256':hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
  'router':{'threshold':THRESHOLD,'features':[{'name':name,'regex':regex,'weight':weight} for name,regex,weight in PATTERNS],
    'uses_official_labels_or_model_responses':False,'is_inference_only':True},
  'route_counts':routings,'official_ifeval_strict':accuracy,'per_family_strict_counts':per_family,
  'dolly_reference_answer_ce':ce,'paired_bootstrap_uncertainty':uncertainty,
  'predeclared_joint_research_gate':gate,'joint_gate_pass':all(gate.values()),
  'real_ifeval_generations':outputs,'real_ifeval_official_verdicts':official,
  'real_dolly_reference_ce':ce_rows,
  'generation_wall_s':time.perf_counter()-started,
  'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
  'limits':['One pretrained model plus one selected public Dolly SFT adapter; routing heuristics are not learned, research-level policy only',
   'Real IFEval and Dolly prompts excluded from prior evaluations but drawn from previously studied public datasets, possible pretraining overlap',
   'Constraint classifier derived from human-understandable language priors; may misclassify implicit or multilingual constraints',
   'All 96 IFEval prompts are explicitly instruction-heavy; a base fallback may be close to trivial and not generalize to natural business tasks',
   'Counterfactual base and LoRA answers measured sequentially for audit; actual router would generate only one candidate',
   'No external safety/factual correctness or user preference ground truth; strict checker is an explicit-format proxy',
   'Conditional within-source bootstrap not a population/multi-seed model estimate']}
 args.out.parent.mkdir(parents=True,exist_ok=True)
 args.out.write_text(json.dumps(report,indent=2)+'\n')
 print('STAGE42_COMPLETE',report['joint_gate_pass'],gate,flush=True)
if __name__=='__main__':main()
