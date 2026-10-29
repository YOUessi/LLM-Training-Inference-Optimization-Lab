"""Build 24 new licensed Python docs requests in 4 REAL Qwen3 token length buckets."""
from __future__ import annotations
import hashlib,importlib,inspect,json
from pathlib import Path

MODULES=('ast','asyncio','bisect','heapq','difflib','base64','binascii','uuid','secrets',
 'os','sys','platform','glob','fnmatch','enum','dataclasses','textwrap','re','math',
 'calendar','gzip','bz2','lzma','zoneinfo')
LENGTHS=(350,700,1100,1550)
INSTRUCTIONS={'label':'Name the main Python standard library module discussed in the passage, in a short noun phrase. No explanation.',
 'summary':'Summarize the source documentation in two sentences.',
 'detailed':'Explain the primary function of this Python standard library module, describe two use cases and one potential limitation. Use only the supplied context and write around 90 to 120 words.'}
# Repeated task header creates an honest shared-prefix-cache regime (not distinct user prompts).
SHARED_HEADER=('You are a software engineer reading the Python documentation. Interpret the reference section carefully. '
               'Use only material written in the provided reference text. Distinguish API behavior from guesses. '
               'Report the essential module purpose and practical restrictions; do not add unsupported implementation details.\n')*4

def build(root:Path)->dict:
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 module_docs={}
 for name in MODULES:
  mod=importlib.import_module(name)
  fragments=[]
  if inspect.getdoc(mod):fragments.append(inspect.getdoc(mod))
  for member in sorted(dir(mod)):
   if member.startswith('_'):continue
   obj=getattr(mod,member,None)
   if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.isbuiltin(obj):
    d=inspect.getdoc(obj)
    if d and len(d)>60:fragments.append(member+': '+d)
   if sum(len(s) for s in fragments)>23000:break
  if not fragments:raise ValueError('no docs '+name)
  module_docs[name]='\n'.join(fragments)
  print('SOURCE',name,'source_chars',len(module_docs[name]))
 rows_by_cohort={mode:[] for mode in ('shared','distinct')}
 for mode in ('shared','distinct'):
  for i,name in enumerate(MODULES):
   target=LENGTHS[i%4]
   head=SHARED_HEADER if mode=='shared' else 'Technical reference for '+name+' only.\n'
   lead=f'CPython module under review: {name}.\n'
   tail='\nQuestion: Which software tasks does the identified module support, and what are the important restrictions?'
   filler=module_docs[name]
   # If too short, add distinct reference API entries from adjacent stdlib modules,
   # explicitly identified as secondary documentation (not claim about target module).
   j=1
   while len(tokenizer.encode(head+lead+filler+tail,add_special_tokens=False))<target+80 and j<len(MODULES):
    sibling=MODULES[(i+j)%len(MODULES)]
    filler+='\nSecondary reference module '+sibling+':\n'+module_docs[sibling]
    j+=1
   head_tokens=len(tokenizer.encode(head+lead+tail,add_special_tokens=False))
   budget=target-head_tokens
   if budget<=20:raise ValueError('too long prefix')
   clipped=tokenizer.decode(tokenizer.encode(filler,add_special_tokens=False)[:budget],skip_special_tokens=True)
   prompt=head+lead+clipped+tail
   cal={}
   for output in ('label','detailed'):
    val=prompt+'\nTask id: 0000. '+INSTRUCTIONS[output]
    cal[output]=len(tokenizer.apply_chat_template([{'role':'user','content':val}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
   if max(cal.values())>1900:raise ValueError('context exceeds safe upper bound '+str(cal))
   item={'request_id':f's25-{mode}-{i:02d}-{name}','language':'en','topic':'stdlib_documentation',
     'bucket':('short','medium','long','very_long')[i%4], 'target_prompt_tokens':cal['detailed'],
     'calibrated_prompt_tokens':cal,'prompt_text':prompt,
     'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(), 'source_module':name}
   rows_by_cohort[mode].append(item)
 manifests={}
 for mode,rows in rows_by_cohort.items():
  blob=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rows).encode()
  path=root/f'stage25_{mode}_requests.jsonl';path.write_bytes(blob)
  manifest={'schema':'stage25-psf-stdlib-v1','cohort':mode,'request_count':24,
    'rights':'Python Software Foundation License (PSF License v2)',
    'attribution':'Python Software Foundation / locally installed CPython module docstrings',
    'generation_instructions':INSTRUCTIONS,'source_modules':list(MODULES),
    'request_jsonl_sha256':hashlib.sha256(blob).hexdigest(),
    'source_kind':'new stdlib modules not used in Stage22; deterministic prompt construction',
    'source_hash_by_module':{n:hashlib.sha256(doc.encode()).hexdigest() for n,doc in module_docs.items()},
    'limitations':['Synthetic prompt framing of real PSF documentation','Shared-prefix cohort intentionally reuses identical header; control does not',
                   'Model may have encountered these docs during pretraining; not production workload']}
  (root/f'stage25_{mode}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  manifests[mode]=manifest
  print('FROZEN',mode,len(rows),manifest['request_jsonl_sha256'],
        'prompt_tokens',[(r['bucket'],r['calibrated_prompt_tokens']['detailed']) for r in rows[:4]])
 return manifests
if __name__=='__main__':build(Path('data'))
