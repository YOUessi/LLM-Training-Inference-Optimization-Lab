"""Freeze another 128 Dolly records disjoint from ALL previous 832 selected IDs.

Build without reading ANY Stage38/40 model scores; evaluates later selected
multiobjective candidate under independent record identities (same Dolly source).
"""
from __future__ import annotations
import collections,hashlib,json
from pathlib import Path
ORIGINAL_SHA='2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec'
SALT='stage41-20261030-new128-no-prior-832'

def build(data:Path,source:Path):
 from transformers import AutoTokenizer
 if hashlib.sha256(source.read_bytes()).hexdigest()!=ORIGINAL_SHA:raise ValueError('original upstream Dolly changed')
 old=set()
 for split in ('train','validation','test'):
  old.update(json.loads(s)['identity_sha256'] for s in (data/f'stage26_{split}.jsonl').read_text().splitlines() if s.strip())
 old.update(json.loads(s)['identity_sha256'] for s in (data/'stage34_new_holdout.jsonl').read_text().splitlines() if s.strip())
 if len(old)!=832:raise ValueError('previous Stage26+34 832 identities changed')
 tok=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 from scripts.build_stage26_dolly_sft import normalize
 buckets=collections.defaultdict(list);seen=set();exclusions=collections.Counter()
 for i,line in enumerate(source.read_text().splitlines()):
  r=json.loads(line)
  instruction,context,response=(r[k].strip() for k in ('instruction','context','response'))
  key=hashlib.sha256((normalize(instruction)+'\n'+normalize(context)).encode()).hexdigest()
  if key in old:exclusions['prior832']+=1;continue
  if key in seen:exclusions['duplicate']+=1;continue
  seen.add(key)
  if not instruction or not response:exclusions['empty']+=1;continue
  user=instruction+('\n\nReference context:\n'+context if context else '')
  start=tok.apply_chat_template([{'role':'user','content':user}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  full=tok.apply_chat_template([{'role':'user','content':user},{'role':'assistant','content':response}],tokenize=True,
    add_generation_prompt=False,enable_thinking=False)
  if len(full)>384 or full[:len(start)]!=start or len(full)-len(start)<2:
   exclusions['length']+=1;continue
  buckets[r['category']].append({'identity_sha256':key,'source_index':i,'category':r['category'],
    'instruction':instruction,'context':context,'response':response,
    'input_tokens':len(start),'target_tokens':len(full)-len(start),'total_tokens':len(full)})
 if len(buckets)!=8 or any(len(x)<16 for x in buckets.values()):raise ValueError('not enough distinct Dolly categories')
 final=[]
 for category in sorted(buckets):
  options=sorted(buckets[category],key=lambda r:hashlib.sha256((SALT+r['identity_sha256']).encode()).hexdigest())
  final+=options[:16]
 final.sort(key=lambda r:hashlib.sha256((SALT+'sort'+r['identity_sha256']).encode()).hexdigest())
 if len(final)!=128 or len({r['identity_sha256'] for r in final})!=128 or any(r['identity_sha256'] in old for r in final):
  raise ValueError('frozen Stage41 source not completely untouched')
 blob=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in final).encode()
 (data/'stage41_dolly_holdout128.jsonl').write_bytes(blob)
 manifest={'schema':'stage41-disjoint-dolly-128-v1','license':'CC BY-SA 3.0 Databricks Dolly15k',
  'source_sha256':ORIGINAL_SHA,'prior832_excluded_count':len(old),
  'selection':'16 each of eight categories by fixed SHA256, 384 complete chat tokens, after excluding prior Stage26 704 and Stage34 128',
  'frozen_input_sha256':hashlib.sha256(blob).hexdigest(),'records':128,
  'category_counts':dict(collections.Counter(x['category'] for x in final)),
  'exclusions':dict(exclusions),'no_model_outcomes_read':True,
  'caveats':['Disjoint record identities but SAME public Dolly source, possible model pretraining overlap',
     'Answer-only CE is not factual or verifiable instruction constraint compliance',
     'Never select new model or its scale from this holdout']}
 (data/'stage41_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('STAGE41_NEW128_FROZEN',manifest['frozen_input_sha256'],manifest['category_counts'])
if __name__=='__main__':build(Path('data'),Path('results/stage26_dolly_source/databricks-dolly-15k.jsonl'))
