"""Freeze 128 *previously unused* Dolly examples for independent post-selection CE."""
from __future__ import annotations
import hashlib,json,collections
from pathlib import Path
SOURCE_SHA='2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec'

def ident(r):
 return hashlib.sha256((' '.join(r['instruction'].lower().split())+'\n'+' '.join(r['context'].lower().split())).encode()).hexdigest()

def build(source:Path,data_dir:Path):
 from transformers import AutoTokenizer
 body=source.read_bytes()
 if hashlib.sha256(body).hexdigest()!=SOURCE_SHA:raise ValueError('official source digest mismatch')
 t=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 previous=set()
 for split in ('train','validation','test'):
  previous.update(json.loads(s)['identity_sha256'] for s in (data_dir/f'stage26_{split}.jsonl').read_text().splitlines() if s.strip())
 if len(previous)!=704:raise ValueError('prior training/test identity count changed')
 groups={};seen=set();excluded=collections.Counter()
 for i,line in enumerate(body.decode().splitlines()):
  r=json.loads(line)
  key=ident(r)
  if key in previous:excluded['previous_stage26_704']+=1;continue
  if key in seen:excluded['duplicate']+=1;continue
  seen.add(key)
  prompt=r['instruction'].strip()+('\n\nReference context:\n'+r['context'].strip() if r['context'].strip() else '')
  chat=[{'role':'user','content':prompt}]
  a=t.apply_chat_template(chat,tokenize=True,add_generation_prompt=True,enable_thinking=False)
  b=t.apply_chat_template(chat+[{'role':'assistant','content':r['response'].strip()}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
  if len(b)>384 or len(b)-len(a)<2 or b[:len(a)]!=a:excluded['token_length']+=1;continue
  groups.setdefault(r['category'],[]).append({'identity_sha256':key,'source_index':i,'category':r['category'],
    'instruction':r['instruction'].strip(),'context':r['context'].strip(),'response':r['response'].strip(),
    'input_tokens':len(a),'target_tokens':len(b)-len(a),'total_tokens':len(b)})
 if len(groups)!=8 or any(len(x)<16 for x in groups.values()):raise ValueError('insufficient qualified source category')
 selected=[]
 for category in sorted(groups):
  order=sorted(groups[category],key=lambda x:hashlib.sha256(('stage34-untouched-20261030'+x['identity_sha256']).encode()).hexdigest())
  selected.extend(order[:16])
 selected.sort(key=lambda x:hashlib.sha256(('stage34-order'+x['identity_sha256']).encode()).hexdigest())
 if len(selected)!=128 or len({x['identity_sha256'] for x in selected})!=128:raise ValueError('not 128 fresh unique holdout')
 if any(r['identity_sha256'] in previous for r in selected):raise ValueError('new test leaked from previous experiment')
 result=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in selected).encode()
 file=data_dir/'stage34_new_holdout.jsonl';file.write_bytes(result)
 manifest={'status':'frozen','schema':'stage34-new-dolly-records-v1','source':'Databricks Dolly 15k',
  'source_original_sha256':SOURCE_SHA,'rights':'CC BY-SA 3.0, Copyright Databricks Inc. (2023)',
  'source_url':'https://huggingface.co/datasets/databricks/databricks-dolly-15k',
  'selection':'16 records per category, SHA256 stage34 salt after excluding all prior Stage26 704 records and duplicates',
  'input_sha256':hashlib.sha256(result).hexdigest(),
  'record_count':128,'category_counts':dict(collections.Counter(r['category'] for r in selected)),
  'prior_704_excluded_count':len(previous),'source_filter_counts':dict(excluded),
  'token_limit':384,'frozen_before_model_evaluation':True,
  'limitations':['New holdout examples from same Dolly source, not an independent domain',
    'Model pretraining may already include this public dataset','Only reference CE, not human semantic correctness']}
 (data_dir/'stage34_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('STAGE34_NEW_UNSEEN_DOLLY128_FROZEN',manifest['input_sha256'],manifest['category_counts'])
 return manifest
if __name__=='__main__':build(Path('results/stage26_dolly_source/databricks-dolly-15k.jsonl'),Path('data'))
