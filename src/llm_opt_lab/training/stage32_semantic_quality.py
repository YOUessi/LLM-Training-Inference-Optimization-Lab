"""Independent mDeBERTa-v3 XNLI semantic-proxy analysis of three frozen 8B models.

Generate all 64 unique prompts (8 per Dolly category) with base, Stage26 short
SFT and validation-selected Stage31 full-SFT, without refit or test selection.
After GPU generation, score both NLI directions separately on CPU.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random,re,statistics,time
from pathlib import Path

MODEL='/home/you/models/Qwen3-8B'
JUDGE='/home/you/.cache/huggingface/hub/models--MoritzLaurer--mDeBERTa-v3-base-mnli-xnli/snapshots/8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c'
MODELS=('base','stage26_short','stage31_full')


def frozen_subset(input_root:Path):
 manifest=json.loads((input_root/'stage26_manifest.json').read_text())
 body=(input_root/'stage26_test.jsonl').read_bytes()
 if hashlib.sha256(body).hexdigest()!=manifest['splits']['test']['sha256']:raise ValueError('frozen test checksum mismatch')
 records=[json.loads(s) for s in body.decode().splitlines()]
 groups={}
 for item in records:groups.setdefault(item['category'],[]).append(item)
 if len(groups)!=8 or any(len(x)!=12 for x in groups.values()):raise ValueError('unexpected category stratification')
 selected=[]
 for category,items in sorted(groups.items()):
  items=sorted(items,key=lambda r:hashlib.sha256(('stage32-fixed-20261030'+r['identity_sha256']).encode()).hexdigest())
  selected.extend(items[:8])
 if len(selected)!=64 or len({x['identity_sha256'] for x in selected})!=64:raise ValueError('duplicate output targets')
 return selected,manifest

def words(s):return re.findall(r"\w+(?:['’-]\w+)?",s.lower(),flags=re.UNICODE)

def lexical(ref,cand):
 from collections import Counter
 a,b=Counter(words(ref)),Counter(words(cand));common=sum((a&b).values());p=common/max(1,sum(b.values()));r=common/max(1,sum(a.values()))
 f1=2*p*r/(p+r) if p+r else 0.
 return {'word_f1':f1,'exact':int(' '.join(words(ref))==' '.join(words(cand))),'output_words':len(words(cand))}

def generate(args):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from peft import PeftModel
 rows,manifest=frozen_subset(args.input)
 selected=json.loads((args.stage31/'selection.json').read_text())
 name=selected['selected'];adapter=args.stage31/(name+'_adapter')
 if not adapter.is_dir():raise FileNotFoundError('validation-selected full-data adapter absent')
 if not args.short_adapter.is_dir():raise FileNotFoundError('Stage26 benchmark adapter absent')
 torch.manual_seed(3232);torch.cuda.manual_seed_all(3232)
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,device_map={'':0},dtype=torch.bfloat16,
  quantization_config=quant,attn_implementation='sdpa')
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 model=PeftModel.from_pretrained(base,args.short_adapter,adapter_name='stage26',is_trainable=False)
 model.load_adapter(adapter,adapter_name='stage31',is_trainable=False)
 model.eval()
 a_sha=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
 b_sha=hashlib.sha256((args.short_adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
 cases=[];start=time.perf_counter()
 with torch.inference_mode():
  for i,row in enumerate(rows):
   user=row['instruction']+('\n\nReference context:\n'+row['context'] if row['context'] else '')
   prompt=tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
   ids=torch.tensor([prompt],device='cuda');mask=torch.ones_like(ids)
   texts={};timings={}
   for name in MODELS:
    begun=time.perf_counter()
    if name=='base':
     with model.disable_adapter():
      outputs=model.generate(ids,attention_mask=mask,max_new_tokens=128,do_sample=False,
       pad_token_id=tokenizer.eos_token_id)
    else:
     model.set_adapter('stage26' if name=='stage26_short' else 'stage31')
     outputs=model.generate(ids,attention_mask=mask,max_new_tokens=128,do_sample=False,
       pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    timings[name]=time.perf_counter()-begun
    texts[name]=tokenizer.decode(outputs[0,len(prompt):],skip_special_tokens=True)
   cases.append({'index':i,'identity_sha256':row['identity_sha256'],'category':row['category'],
    'reference_sha256':hashlib.sha256(row['response'].encode()).hexdigest(),
    'outputs':texts,'generation_s':timings,
    'generation_sha256':{m:hashlib.sha256(text.encode()).hexdigest() for m,text in texts.items()},
    'lexical':{m:lexical(row['response'],text) for m,text in texts.items()}})
   if (i+1)%8==0:print('STAGE32_GPU_GENERATED',i+1,'/',len(rows),flush=True)
 result={'status':'measured','kind':'stage32_64_frozen_three_model_greedy_generations',
  'test_source_sha256':manifest['splits']['test']['sha256'],'source_dataset_sha256':manifest['source_sha256'],
  'selected_stage31':selected['selected'],
  'selected_stage31_rank':selected['rank'],'short_adapter_sha256':b_sha,'full_adapter_sha256':a_sha,
  'chosen_id_rule':'first 8 per category under fixed SHA256 rank; no output-based cherry picking',
  'config':{'max_new_tokens':128,'thinking':False,'greedy':True,'models':list(MODELS),'generated_count':64},
  'gpu_generation_elapsed_s':time.perf_counter()-start,'cases':cases,
  'limitations':['Reference-based lexical overlap alone does not validate semantic/factual answer correctness',
   'Greedy decoding at fixed 128-token cap can truncate detailed answers',
   'Test source may overlap model pretraining, no human preference or independent manual annotation']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
 print('STAGE32_GENERATION_COMPLETE',len(cases),'x',len(MODELS),flush=True)

def score_nli(args):
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 torch.set_num_threads(5)
 x=json.loads(args.generated.read_text())
 if x['status']!='measured' or len(x['cases'])!=64:raise ValueError('missing generated comparison')
 rows,manifest=frozen_subset(args.input)
 if x['test_source_sha256']!=manifest['splits']['test']['sha256']:raise ValueError('test content hash drift')
 expected={r['identity_sha256']:r for r in rows}
 tokenizer=AutoTokenizer.from_pretrained(JUDGE,local_files_only=True)
 judge=AutoModelForSequenceClassification.from_pretrained(JUDGE,local_files_only=True).cpu().eval()
 labs={label.lower():int(i) for i,label in judge.config.id2label.items()}
 if set(labs)!={'entailment','neutral','contradiction'}:raise ValueError('unsupported NLI label map')
 premises=[];hypotheses=[];metadata=[]
 for r in x['cases']:
  ref=expected[r['identity_sha256']]['response']
  if hashlib.sha256(ref.encode()).hexdigest()!=r['reference_sha256']:raise ValueError('reference SHA changed')
  for name in MODELS:
   text=r['outputs'][name]
   if hashlib.sha256(text.encode()).hexdigest()!=r['generation_sha256'][name]:raise ValueError('generated text drift')
   premises.extend([ref,text]);hypotheses.extend([text,ref]);metadata.extend([(r['identity_sha256'],name,'reference_to_candidate'),(r['identity_sha256'],name,'candidate_to_reference')])
 predictions=[]
 with torch.inference_mode():
  for i in range(0,len(premises),12):
   tokens=tokenizer(premises[i:i+12],hypotheses[i:i+12],truncation=True,max_length=384,padding=True,return_tensors='pt')
   scores=judge(**tokens).logits.softmax(dim=-1).tolist()
   predictions.extend(scores)
 if len(predictions)!=64*3*2:raise ValueError('insufficient semantic pairs')
 lookup={}
 for (identity,name,direction),probs in zip(metadata,predictions):
  if len(probs)!=3:raise ValueError('invalid NLI class scores')
  lookup.setdefault(identity,{}).setdefault(name,{})[direction]={k:probs[i] for k,i in labs.items()}
 cases=[]
 for rec in x['cases']:
  entry={'identity_sha256':rec['identity_sha256'],'category':rec['category'],'metrics':{}}
  for name in MODELS:
   nli=lookup[rec['identity_sha256']][name]
   entry['metrics'][name]={'lexical':rec['lexical'][name],
    'reference_entails_candidate':nli['reference_to_candidate']['entailment'],
    'candidate_entails_reference':nli['candidate_to_reference']['entailment'],
    'reference_contradicts_candidate':nli['reference_to_candidate']['contradiction'],
    'mutual_entailment_mean':.5*(nli['reference_to_candidate']['entailment']+nli['candidate_to_reference']['entailment'])}
  cases.append(entry)
 def summarize(subset):
  return {m:{k:statistics.mean(z['metrics'][m][k] if k!='word_f1' else z['metrics'][m]['lexical']['word_f1'] for z in subset)
    for k in ('reference_entails_candidate','candidate_entails_reference','reference_contradicts_candidate','mutual_entailment_mean','word_f1')}
    for m in MODELS}
 overall=summarize(cases)
 cats={name:summarize([x for x in cases if x['category']==name]) for name in sorted({x['category'] for x in cases})}
 rng=random.Random(320032)
 # Resample 8 samples per category in each bootstrap replicate to preserve category balance.
 def interval_metric(metric,against):
  samples=[];categories={cat:[r for r in cases if r['category']==cat] for cat in cats}
  for _ in range(2000):
   drawn=[]
   for category in sorted(categories):
    rows=categories[category]
    drawn.extend(rows[rng.randrange(len(rows))] for _ in range(len(rows)))
   delta=statistics.mean(r['metrics']['stage31_full'][metric]-r['metrics'][against][metric] for r in drawn)
   samples.append(delta)
  samples.sort();return [samples[50],samples[1949]]
 uncertainty={against:{metric:interval_metric(metric,against) for metric in ('reference_entails_candidate','mutual_entailment_mean')}
   for against in ('base','stage26_short')}
 out={'status':'measured','kind':'stage32_64_stratified_independent_xnli_semantic_proxy',
  'judge_checkpoint':JUDGE,'judge_id2label':judge.config.id2label,
  'judge_task':'Natural-language inference, not human factual evaluation',
  'model_generations_sha256':hashlib.sha256(args.generated.read_bytes()).hexdigest(),
  'test_source_sha256':manifest['splits']['test']['sha256'],'cases':cases,'counts':{'items':64,'models':3,'nli_directional_pairs':len(predictions)},
  'overall':overall,'per_category':cats,'stratified_bootstrap_95pct_ci_deltas_new_vs':uncertainty,
  'limitations':['XNLI entailment is a FALLIBLE out-of-domain surrogate, especially open-ended creative writing',
    'Reference entails candidate is not verified evidence grounding or factuality',
    'Four methods share public Dolly test may appear in model pretraining',
    'Stratified bootstrap captures within-fixed-test sampling only, not model training randomness',
    'Long answers may be truncated by NLI token cap 384 and 128-token generation cap']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+'\n')
 print('STAGE32_NLI_COMPLETE',len(cases),'nli_pairs',len(predictions),flush=True)
 for m,score in overall.items():print('MODEL',m,'NLI_MEAN',round(score['mutual_entailment_mean'],4),'CONTRADICTION',round(score['reference_contradicts_candidate'],4),'TOKEN_F1',round(score['word_f1'],4),flush=True)
 print('STRATIFIED_CIS',uncertainty,flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--action',choices=('generate','nli'),required=True)
 p.add_argument('--input',type=Path,default=Path('data'))
 p.add_argument('--stage31',type=Path,default=Path('results/stage31_full512_two_epochs'))
 p.add_argument('--short-adapter',type=Path,default=Path('evidence/stage26_sft_v2/selected_B_adapter'))
 p.add_argument('--generated',type=Path)
 p.add_argument('--out',type=Path,required=True)
 args=p.parse_args()
 if args.action=='generate':generate(args)
 else:
  if not args.generated:raise ValueError('--generated required for NLI')
  score_nli(args)
if __name__=='__main__':main()
