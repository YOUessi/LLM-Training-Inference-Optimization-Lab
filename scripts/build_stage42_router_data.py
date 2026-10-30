"""Preregister new IFEval96 and disjoint Dolly128: no previous test key reused.

Neither the router nor model outcome enters example selection.
"""
from __future__ import annotations
import hashlib,json,collections
from pathlib import Path
UPSTREAM_IFEVAL_SHA='6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2'
UPSTREAM_DOLLY_SHA='2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec'
SALT='stage42-2026-10-30-independent-source-identities-anti-leak-v1'
QUOTAS={'detectable_content':4,'detectable_format':30,'length_constraints':8,'combination':8,
        'change_case':10,'keywords':30,'punctuation':2,'startend':2,'language':2}

def build(data:Path,ifeval_source:Path,dolly_source:Path):
 from transformers import AutoTokenizer
 from build_stage36_ifeval import eligible
 if hashlib.sha256(ifeval_source.read_bytes()).hexdigest()!=UPSTREAM_IFEVAL_SHA:raise ValueError('IFEval official source changed')
 if hashlib.sha256(dolly_source.read_bytes()).hexdigest()!=UPSTREAM_DOLLY_SHA:raise ValueError('Dolly official source changed')
 old_ifeval=set()
 for name in ('stage36_ifeval_96.jsonl','stage38_ifeval_dev.jsonl','stage38_ifeval_test.jsonl',
              'stage40_ifeval_dev.jsonl','stage40_ifeval_test.jsonl'):
  old_ifeval|={json.loads(s)['key'] for s in (data/name).read_text().splitlines() if s.strip()}
 if len(old_ifeval)!=325:raise ValueError('expected 325 prior examined IFEval identities')
 tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 groups=collections.defaultdict(list)
 for line in ifeval_source.read_text().splitlines():
  r=json.loads(line)
  if r['key'] in old_ifeval or not eligible(r)[0]:continue
  tokens=len(tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
  if tokens+384>2048:continue
  family=r['instruction_id_list'][0].split(':')[0]
  groups[family].append(r)
 if len(groups)!=len(QUOTAS) or sum(QUOTAS.values())!=96:raise ValueError('unregistered family quotas')
 final=[]
 for family,n in QUOTAS.items():
  candidates=sorted(groups[family],key=lambda r:hashlib.sha256((SALT+'-ifeval-'+str(r['key'])).encode()).hexdigest())
  if len(candidates)<n:raise ValueError('unavailable IFEval category '+family)
  final.extend(candidates[:n])
 final.sort(key=lambda r:hashlib.sha256((SALT+'-ifeval-order-'+str(r['key'])).encode()).hexdigest())
 if len(final)!=96 or len({r['key'] for r in final})!=96 or any(r['key'] in old_ifeval for r in final):
  raise ValueError('IFEval data leakage or duplicate')
 ibytes=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in final).encode()
 (data/'stage42_ifeval_untouched96.jsonl').write_bytes(ibytes)
 prior_dolly=set()
 for name in ('stage26_train.jsonl','stage26_validation.jsonl','stage26_test.jsonl',
              'stage34_new_holdout.jsonl','stage41_dolly_holdout128.jsonl'):
  prior_dolly|={json.loads(x)['identity_sha256'] for x in (data/name).read_text().splitlines() if x.strip()}
 if len(prior_dolly)!=960:raise ValueError('expected exactly 960 already examined Dolly identities')
 from build_stage26_dolly_sft import normalize
 categories=collections.defaultdict(list);seen=set();filtered=collections.Counter()
 for idx,line in enumerate(dolly_source.read_text().splitlines()):
  r=json.loads(line)
  instruction,context,response=(r[k].strip() for k in ('instruction','context','response'))
  identity=hashlib.sha256((normalize(instruction)+'\n'+normalize(context)).encode()).hexdigest()
  if identity in prior_dolly:filtered['prior_960']+=1;continue
  if identity in seen:filtered['duplicate']+=1;continue
  seen.add(identity)
  if not instruction or not response:filtered['empty']+=1;continue
  user=instruction+('\n\nReference context:\n'+context if context else '')
  prefix=tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  full=tokenizer.apply_chat_template([{'role':'user','content':user},{'role':'assistant','content':response}],
     tokenize=True,add_generation_prompt=False,enable_thinking=False)
  if len(full)>384 or full[:len(prefix)]!=prefix or len(full)-len(prefix)<2:
   filtered['token_limit']+=1;continue
  categories[r['category']].append({'identity_sha256':identity,'source_index':idx,'category':r['category'],
     'instruction':instruction,'context':context,'response':response,
     'prompt_tokens':len(prefix),'full_tokens':len(full),'target_tokens':len(full)-len(prefix)})
 if len(categories)!=8 or any(len(v)<16 for v in categories.values()):raise ValueError('Dolly 8 categories insufficient')
 drows=[]
 for family in sorted(categories):
  options=sorted(categories[family],key=lambda r:hashlib.sha256((SALT+'-dolly-'+r['identity_sha256']).encode()).hexdigest())
  drows.extend(options[:16])
 drows.sort(key=lambda r:hashlib.sha256((SALT+'-dolly-order-'+r['identity_sha256']).encode()).hexdigest())
 if len(drows)!=128 or len({r['identity_sha256'] for r in drows})!=128 or any(r['identity_sha256'] in prior_dolly for r in drows):
  raise ValueError('Dolly 128 source identity collision')
 dbytes=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in drows).encode()
 (data/'stage42_dolly_untouched128.jsonl').write_bytes(dbytes)
 manifest={'schema':'stage42-clean-ifeval96-dolly128-fixed-20261030-v1',
    'source_ifeval_sha256':UPSTREAM_IFEVAL_SHA,'source_dolly_sha256':UPSTREAM_DOLLY_SHA,
    'licensing':{'IFEval':'Apache-2.0 Google Research','Dolly':'CC BY-SA 3.0 Databricks Inc.'},
    'ifeval':{'count':96,'sha256':hashlib.sha256(ibytes).hexdigest(),'prior_325_excluded':325,
       'category_counts':dict(collections.Counter(r['instruction_id_list'][0].split(':')[0] for r in final)),
       'keys':[r['key'] for r in final],'fixed_quota':QUOTAS},
    'dolly':{'count':128,'sha256':hashlib.sha256(dbytes).hexdigest(),'prior_960_excluded':960,
       'category_counts':dict(collections.Counter(r['category'] for r in drows)),
       'excluded_by_reason':dict(filtered)},
    'selection':'SHA256 deterministic tier quotas, no model/adapter scores consulted, original official sources only',
    'model_checkpoints':'Original Qwen3-8B and validation-selected Stage31 rank16 QLoRA only',
    'data_publication':'All exact IFEval and Dolly user/answer text originally in public licensed sources; no OASST personal user data',
    'limits':['Same IFEval source family despite unused individual examples, not true independent task distribution',
              'Dolly new IDs are same Dolly public source, may have entered Qwen3 pretraining',
              'One model/adapter, observational fixed-data randomization, no human quality labels']}
 (data/'stage42_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('STAGE42_UNSEEN_FROZEN','IFEVAL',len(final),manifest['ifeval']['sha256'],'DOLLY',len(drows),manifest['dolly']['sha256'])
 return manifest
if __name__=='__main__':
 build(Path('data'),Path('results/stage35_public_data/ifeval_input_data.jsonl'),Path('results/stage26_dolly_source/databricks-dolly-15k.jsonl'))
