"""License-audited deterministic Dolly instruction-response split; answer-only loss."""
from __future__ import annotations
import collections,hashlib,json,random,re
from pathlib import Path

SOURCE_SHA='2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec'
MAX_LEN=384
SPLITS={'train':64,'validation':12,'test':12}
SEED=2601029

def normalize(s:str)->str:
 return ' '.join(s.lower().split())

def convert(source:Path,output:Path):
 from transformers import AutoTokenizer
 data=source.read_bytes()
 if hashlib.sha256(data).hexdigest()!=SOURCE_SHA:
  raise ValueError('untrusted upstream dataset checksum')
 t=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 rows=[json.loads(line) for line in data.decode().splitlines() if line.strip()]
 if len(rows)!=15011:raise ValueError('upstream dataset record-count drift')
 categories=sorted({x['category'] for x in rows})
 if len(categories)!=8:raise ValueError('expected 8 official categories')
 buckets={c:[] for c in categories}
 seen=set();reasons=collections.Counter()
 for i,r in enumerate(rows):
  if set(r)!={'instruction','context','response','category'}:raise ValueError('unexpected upstream schema')
  instruction,context,response=(r[k].strip() for k in ('instruction','context','response'))
  category=r['category']
  if not instruction or not response:reasons['empty']+=1;continue
  if any(s in (instruction+context+response).lower() for s in ('sk-ws-', 'ghp_', '-----begin private key-----')):
   reasons['possible_secret']+=1;continue
  key=hashlib.sha256((normalize(instruction)+'\n'+normalize(context)).encode()).hexdigest()
  if key in seen:reasons['duplicate_instruction_context']+=1;continue
  seen.add(key)
  user=instruction+('\n\nReference context:\n'+context if context else '')
  msg=[{'role':'user','content':user}]
  prefix=t.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False)
  full=t.apply_chat_template(msg+[{'role':'assistant','content':response}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
  if len(prefix)>240 or len(full)>MAX_LEN or len(full)-len(prefix)<2:
   reasons['token_length']+=1;continue
  if prefix!=full[:len(prefix)]:raise RuntimeError('prompt masking boundary inconsistency')
  buckets[category].append({'source_index':i,'source_sha256':hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest(),
      'identity_sha256':key,'category':category,'instruction':instruction,'context':context,
      'response':response,'input_tokens':len(prefix),'target_tokens':len(full)-len(prefix),
      'total_tokens':len(full)})
 output.mkdir(parents=True,exist_ok=True)
 counts={};splits={k:[] for k in SPLITS}
 for c in categories:
  choices=buckets[c]
  if len(choices)<sum(SPLITS.values()):raise ValueError(f'insufficient real records for category {c}: {len(choices)}')
  order=sorted(choices,key=lambda r:hashlib.sha256((str(SEED)+r['identity_sha256']).encode()).hexdigest())
  j=0
  for split,n in SPLITS.items():
   splits[split].extend(order[j:j+n]);j+=n
  counts[c]={'qualified_records':len(choices),'selected_each':SPLITS.copy()}
 output_meta={}
 identities=[]
 for split,subset in splits.items():
  subset=sorted(subset,key=lambda r:hashlib.sha256((split+str(SEED)+r['identity_sha256']).encode()).hexdigest())
  identities.extend(r['identity_sha256'] for r in subset)
  content=''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n' for row in subset)
  path=output/f'stage26_{split}.jsonl';path.write_text(content,encoding='utf-8')
  output_meta[split]={'rows':len(subset),'sha256':hashlib.sha256(content.encode()).hexdigest(),
    'per_category':dict(collections.Counter(r['category'] for r in subset)),
    'mean_prompt_tokens':round(sum(r['input_tokens'] for r in subset)/len(subset),2),
    'mean_answer_tokens':round(sum(r['target_tokens'] for r in subset)/len(subset),2)}
 if len(identities)!=len(set(identities)):raise ValueError('cross-split leakage')
 manifest={'schema':'dolly-instruction-sft-v1','dataset':'databricks/databricks-dolly-15k',
  'source_url':'https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl',
  'download_via':'hf-mirror.com verified by official source SHA256','source_sha256':SOURCE_SHA,
  'license':'CC-BY-SA-3.0','attribution':'Copyright (2023) Databricks, Inc.; Wikipedia contributors for relevant referenced contexts',
  'license_url':'https://creativecommons.org/licenses/by-sa/3.0/',
  'model_tokenizer':'/home/you/models/Qwen3-8B (local)',
  'format':'Qwen3 no-thinking chat template; assistant-only loss; complete, not truncated, 384-token max',
  'seed':SEED,'max_sequence_tokens':MAX_LEN,'original_rows':len(rows),
  'category_counts':counts,'splits':output_meta,'filtered_reasons':dict(reasons),
  'holdout_policy':'no training on validation/test; selection by normalized instruction+context identity; fixed before experiment',
  'caveats':['Dataset is public but could overlap pretrained Qwen knowledge','Limited subset and 384-token filter bias sample distribution','No human preference scoring or safety-alignment ground truth']}
 (output/'stage26_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 print('DOLLY_FROZEN_VERIFIED',len(identities),output_meta,'rejected',dict(reasons))
 return manifest
if __name__=='__main__':convert(Path('results/stage26_dolly_source/databricks-dolly-15k.jsonl'),Path('data'))
