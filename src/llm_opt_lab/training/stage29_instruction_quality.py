"""Frozen, paired 96-item reference comparison: base vs already-selected 8B adapter.

This is a post-selection diagnostic: never pick another adapter using test labels.
Scores are lexical overlap and exact-match, not human semantic correctness.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re,statistics,time
from pathlib import Path

def tokenize(text:str)->list[str]:
 return re.findall(r"\w+(?:['’-]\w+)?",text.lower(),flags=re.UNICODE)

def quality(reference:str,generated:str)->dict:
 from collections import Counter
 rt,gt=tokenize(reference),tokenize(generated)
 a,b=Counter(rt),Counter(gt)
 both=sum((a&b).values());p=both/max(1,len(gt));r=both/max(1,len(rt))
 f1=2*p*r/(p+r) if p+r>0 else 0.
 # ROUGE-L word sequence via O(L*M) dynamic program: words truncated only for metric cost.
 x,y=rt[:1000],gt[:1000]
 previous=[0]*(len(y)+1)
 for t in x:
  current=[0]*(len(y)+1)
  for j,u in enumerate(y,1):current[j]=previous[j-1]+1 if t==u else max(current[j-1],previous[j])
  previous=current
 lcs=previous[-1];pl=lcs/max(1,len(y));rl=lcs/max(1,len(x))
 rouge=2*pl*rl/(pl+rl) if pl+rl else 0.
 return {'token_f1':f1,'rouge_l_f1':rouge,
  'normalized_exact_match':int(' '.join(rt)==' '.join(gt)),
  'prediction_words':len(gt),'reference_words':len(rt)}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--input',type=Path,default=Path('data'))
 ap.add_argument('--adapter',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--limit',type=int,choices=(24,96),default=96)
 args=ap.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from peft import PeftModel
 meta=json.loads((args.input/'stage26_manifest.json').read_text())
 body=(args.input/'stage26_test.jsonl').read_bytes()
 if hashlib.sha256(body).hexdigest()!=meta['splits']['test']['sha256']:raise ValueError('frozen test corpus SHA mismatch')
 rows=[json.loads(s) for s in body.decode().splitlines()]
 if len(rows)!=96:raise ValueError('missing held-out test rows')
 selected_config=json.loads((args.adapter/'adapter_config.json').read_text())
 if selected_config['r']!=16:raise ValueError('must evaluate frozen rank16 validation-selected adapter')
 torch.manual_seed(29029);torch.cuda.manual_seed_all(29029)
 qconf=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 model_dir='/home/you/models/Qwen3-8B'
 base=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,
  device_map={'':0},dtype=torch.bfloat16,quantization_config=qconf,attn_implementation='sdpa')
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
 model=PeftModel.from_pretrained(base,args.adapter,is_trainable=False)
 model.eval()
 t0=time.perf_counter();details=[]
 with torch.inference_mode():
  for i,row in enumerate(rows[:args.limit]):
   user=row['instruction']+('\n\nReference context:\n'+row['context'] if row['context'] else '')
   tokens=tok.apply_chat_template([{'role':'user','content':user}],tokenize=True,
      add_generation_prompt=True,enable_thinking=False)
   x=torch.tensor([tokens],device='cuda')
   attention=torch.ones_like(x)
   before=time.perf_counter()
   with model.disable_adapter():
    base_ids=model.generate(x,attention_mask=attention,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id)
   base_generated=tok.decode(base_ids[0,len(tokens):],skip_special_tokens=True)
   base_latency=time.perf_counter()-before
   before=time.perf_counter()
   tuned_ids=model.generate(x,attention_mask=attention,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id)
   tuned_generated=tok.decode(tuned_ids[0,len(tokens):],skip_special_tokens=True)
   tuned_latency=time.perf_counter()-before
   ref=row['response']
   details.append({'id_sha256':row['identity_sha256'],'category':row['category'],
      'input_tokens':len(tokens),'reference_sha256':hashlib.sha256(ref.encode()).hexdigest(),
      'base':{**quality(ref,base_generated),'sha256':hashlib.sha256(base_generated.encode()).hexdigest(),
              'output_tokens':len(base_ids[0])-len(tokens),'generation_s':base_latency},
      'tuned':{**quality(ref,tuned_generated),'sha256':hashlib.sha256(tuned_generated.encode()).hexdigest(),
               'output_tokens':len(tuned_ids[0])-len(tokens),'generation_s':tuned_latency}})
   if (i+1)%12==0:print('STAGE29_GENERATED',i+1,'/',args.limit,flush=True)
 def summaries(items):
  result={}
  for key in ('base','tuned'):
   result[key]={k:statistics.mean(e[key][k] for e in items) for k in
     ('token_f1','rouge_l_f1','normalized_exact_match','prediction_words','generation_s')}
  result['paired_diff']={k:result['tuned'][k]-result['base'][k] for k in ('token_f1','rouge_l_f1','normalized_exact_match')}
  result['tuned_better_rouge_cases']=sum(e['tuned']['rouge_l_f1']>e['base']['rouge_l_f1'] for e in items)
  result['base_better_rouge_cases']=sum(e['tuned']['rouge_l_f1']<e['base']['rouge_l_f1'] for e in items)
  return result
 categories=sorted({x['category'] for x in details})
 report={'status':'measured','kind':'stage29_96_frozen_dolly_paired_greedy_instruction_quality',
  'dataset_sha256':meta['source_sha256'],'test_split_sha256':meta['splits']['test']['sha256'],
  'adapter_sha256':hashlib.sha256((args.adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
  'generation':{'do_sample':False,'max_new_tokens':128,'enable_thinking':False},
  'total_cases':len(details),'wall_s':time.perf_counter()-t0,
  'overall':summaries(details),
  'per_category':{c:summaries([x for x in details if x['category']==c]) for c in categories},
  'cases':details,
  'limitations':['Paired reference-overlap and ROUGE-L metrics are not semantic correctness or safety human evaluations',
   'Dataset may overlap pretraining; no human preference labels','Selected LoRA was already fixed using only validation; test not used for selection',
   'Deterministic max128 outputs can truncate detailed answers; source references have varying lengths']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
 print('STAGE29_REAL_GUARDED_TEST',json.dumps(report['overall'],indent=2),flush=True)
if __name__=='__main__':main()
