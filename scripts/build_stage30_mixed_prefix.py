"""Freeze 96 unique actual PSF stdlib document questions with paired hot/cold forms."""
from __future__ import annotations
import hashlib,importlib,inspect,json
from pathlib import Path
MODULES=('ast','asyncio','bisect','heapq','difflib','base64','binascii','uuid','secrets','os','sys','platform',
 'glob','fnmatch','enum','dataclasses','textwrap','re','math','calendar','gzip','bz2','lzma','zoneinfo')
TARGETS=(360,680,1010,1380)
SHARED=('This is a technical reference-reading task for a software engineer. Use only the following source. '
        'Keep the distinction between API guarantees, illustrative examples, and unspecified behavior. '
        'Never fabricate performance promises, unsupported API flags or missing error handling details. '
        'The relevant evidence is in the documented source below.\n')*4
INSTRUCTIONS={'label':'Identify the primary Python standard library module named in the source. Reply with only its name.',
 'detailed':'Explain this module with at least six substantial technical points, concrete examples, common edge cases, and limitations. Produce a thorough 180 to 220 word answer; never answer in a short sentence.'}

def build(root:Path):
 from transformers import AutoTokenizer
 t=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 all_docs={}
 for module in MODULES:
  obj=importlib.import_module(module)
  fragments=[]
  if inspect.getdoc(obj):fragments.append(inspect.getdoc(obj))
  for member in sorted(dir(obj)):
   if member.startswith('_'):continue
   thing=getattr(obj,member,None)
   if inspect.isclass(thing) or inspect.isfunction(thing) or inspect.isbuiltin(thing):
    doc=inspect.getdoc(thing)
    if doc and len(doc)>40:fragments.append(member+': '+doc)
   if sum(len(x) for x in fragments)>=14000:break
  if not fragments:raise ValueError('no source for '+module)
  all_docs[module]='\n'.join(fragments)
 rows=[]
 for variant in range(4):
  for i,name in enumerate(MODULES):
   identity=f'stage30-{variant:01d}-{i:02d}-{name}'
   topic=['practical usage','edge cases','API selection','testing and failure handling'][variant]
   target=TARGETS[(i+variant)%4]
   text=all_docs[name]
   k=1
   while len(t.encode(text,add_special_tokens=False))<target+120 and k<len(MODULES):
    neighbor=MODULES[(i+k)%len(MODULES)]
    text+='\nSecondary documented API ('+neighbor+'):\n'+all_docs[neighbor]
    k+=1
   prompt_forms={}; token_counts={}
   for kind in ('hot','cold'):
    header=SHARED if kind=='hot' else ('Unique reference task '+identity+': '+topic+'. This is a separate independent prompt.\n')
    lead=f'Primary Python module: {name}. Context purpose: {topic}.\n'
    tail='\nQuestion: describe what the primary module enables and which limitations matter.'
    count=len(t.encode(header+lead+tail,add_special_tokens=False))
    excerpt=t.decode(t.encode(text,add_special_tokens=False)[:max(60,target-count)],skip_special_tokens=True)
    prompt=header+lead+excerpt+tail
    cal={}
    for mode,instruction in INSTRUCTIONS.items():
     actual=f'{prompt}\nTask id: 0000. {instruction}'
     cal[mode]=len(t.apply_chat_template([{'role':'user','content':actual}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
    if max(cal.values())>1750:raise ValueError('would exceed model context with answer '+str(cal))
    prompt_forms[kind]=prompt
    token_counts[kind]=cal
   rows.append({'request_id':identity,'language':'en','topic':'python_stdlib',
    'bucket':('short','medium','long','very_long')[(i+variant)%4],
    'primary_module':name,'variant':variant,'source_sha256':hashlib.sha256(text.encode()).hexdigest(),
    'prompts':prompt_forms,'calibrated_prompt_tokens_by_prefix':token_counts,
    'prompt_sha256_by_prefix':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in prompt_forms.items()}})
 assert len(rows)==96 and len({r['request_id'] for r in rows})==96
 root.mkdir(parents=True,exist_ok=True)
 body=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode()
 (root/'stage30_requests.jsonl').write_bytes(body)
 manifest={'schema':'stage30-long-window-cold-hot-v1','request_count':96,'request_jsonl_sha256':hashlib.sha256(body).hexdigest(),
    'source':'locally installed CPython PSF-licensed module/member documentation, 24 modules x4 question variants',
    'attribution':'Python Software Foundation, CPython documentation; PSF License 2',
    'license_url':'https://docs.python.org/3/license.html',
    'generation_instructions':INSTRUCTIONS,'source_modules':list(MODULES),'context_targets':list(TARGETS),
    'shared_prefix_sha256':hashlib.sha256(SHARED.encode()).hexdigest(),
    'planned_mixes':{'cold0':0,'hot50':48,'hot100':96},
    'planned_seeds':[3011,3029,3041], 'planned_rate_per_second':8.,
    'mode_counts':{'label':48,'detailed':48},
    'limits':['Deterministic engineered cache-hit regime, not naturally sampled user queries',
              'The source domain is related to prior Stage25 stdlib prompts though request identities are new',
              'Cold form means nonshared prompt prefix, not completely empty GPU/cache: server freshly restarted for every arm']}
 (root/'stage30_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('STAGE30_CORPUS_FROZEN',len(rows),manifest['request_jsonl_sha256'])
 print('TOKEN_EXAMPLE',rows[0]['calibrated_prompt_tokens_by_prefix'])
if __name__=='__main__':build(Path('data'))
