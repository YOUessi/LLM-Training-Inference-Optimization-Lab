"""Freeze a new public PSF-licensed Python stdlib technical-domain benchmark."""
from __future__ import annotations
import hashlib,importlib,inspect,json
from pathlib import Path

MODULES=('pathlib','json','csv','logging','argparse','contextlib','collections','itertools',
         'functools','datetime','decimal','fractions','typing','shutil','tempfile','socket',
         'ipaddress','urllib.parse','http.client','sqlite3','unittest','hashlib','statistics','subprocess')
BUCKETS=('short','medium','long','very_long')
TARGETS=(170,300,470,680)
INSTRUCTIONS={
 'label':'Name only the Python standard-library module described in the source passage, without explanation.',
 'summary':'Explain the module purpose in two clear sentences, using the provided documentation.',
 'detailed':'Give a detailed explanation of the module purpose, suitable use cases, and one caveat, based on the documentation. Write about 100 words.',
}

def build(root:Path)->dict:
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 rows=[]
 for i,modname in enumerate(MODULES):
  mod=importlib.import_module(modname)
  parts=[]
  if inspect.getdoc(mod):parts.append(inspect.getdoc(mod))
  for name in sorted(dir(mod)):
   if name.startswith('_'):continue
   thing=getattr(mod,name,None)
   if inspect.isclass(thing) or inspect.isfunction(thing) or inspect.isbuiltin(thing):
    doc=inspect.getdoc(thing)
    if doc and len(doc)>70:parts.append(name+': '+doc)
   if sum(len(x) for x in parts)>16000:break
  if not parts:raise ValueError('documentation missing '+modname)
  bucket=BUCKETS[i%4];target=TARGETS[i%4]
  source='\n'.join(parts)
  prefix=f'CPython standard library documentation source for module {modname}.\n'
  tail='\nQuestion: What software tasks does this module address? Explain using only the source.'
  # Token-bounded documented content; no model-generated text or labels used.
  budget=max(90,target-len(tokenizer.encode(prefix+tail,add_special_tokens=False))-25)
  content=tokenizer.decode(tokenizer.encode(source,add_special_tokens=False)[:budget],skip_special_tokens=True)
  prompt=prefix+content+tail
  cal={}
  for mode in ('label','detailed'):
   payload=f'{prompt}\nTask id: 0000. {INSTRUCTIONS[mode]}'
   cal[mode]=len(tokenizer.apply_chat_template([{'role':'user','content':payload}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False))
  rows.append(dict(request_id=f'stdlib-{i+1:02d}-{modname.replace(".","-")}',
      language='en',topic='python_stdlib',bucket=bucket,target_prompt_tokens=cal['detailed'],
      prompt_text=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
      calibrated_prompt_tokens=cal,origin_module=modname))
 root.mkdir(parents=True,exist_ok=True)
 body=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rows).encode()
 (root/'stage22_stdlib_requests.jsonl').write_bytes(body)
 manifest=dict(schema='stage22-stdlib-domain-v1',source='locally installed CPython standard-library module and member docstrings',
      attribution='Python Software Foundation, CPython standard-library documentation',
      rights='Python Software Foundation License; attribution retained',
      copyright_notice='See https://docs.python.org/3/license.html',
      data_process='deterministic inspect.getdoc member sampling and tokenizer-token truncation',
      model_tokenizer='Qwen3-8B local',request_count=len(rows),request_jsonl_sha256=hashlib.sha256(body).hexdigest(),
      generation_instructions=INSTRUCTIONS,bucket_counts={b:sum(x['bucket']==b for x in rows) for b in BUCKETS},
      modules=list(MODULES),limitations=['New content domain but documents from local standard library; not externally collected user prompts',
                                        'Output questions are synthetic and module names appear in source'])
 (root/'stage22_stdlib_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return manifest
if __name__=='__main__':
 m=build(Path('data'));print('STAGE22_FROZEN',m['request_count'],m['request_jsonl_sha256'])
