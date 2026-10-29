"""Freeze sanitized natural OpenAssistant root queries with public-only hash manifest.

Unmodified user texts remain on authorized Tang in ignored results; GitHub gets
provenance, anonymized content hashes and code only (no user IDs/raw prompts).
"""
from __future__ import annotations
import hashlib,gzip,json,re,collections
from pathlib import Path
SOURCE='621ccd86a6ef320ca4e24c137121bd4b39bcc7a0df839f0897fcc965ef2076ed'
LANGS=('en','es','zh')
SYSTEM='You are an assistant helping a user with a genuine request. Respond in the same language as the user. '
SYSTEM+=('Prefer clear answers, truthful statements, useful examples and a considerate tone. '
 'If information is missing, be transparent about uncertainty; avoid invented facts. '
 'Keep the response focused on the user request and do not disclose private data. ')*4
BAD=re.compile(r'(?i)(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://\S+|www\.\S+|\+?\d[\d\- ()]{8,}\d|-----BEGIN|sk-[\w\-]{12,}|ghp_[A-Za-z0-9]+)')

def build(src:Path,private_dir:Path,public_dir:Path):
 from transformers import AutoTokenizer
 if hashlib.sha256(src.read_bytes()).hexdigest()!=SOURCE:raise ValueError('upstream public source integrity mismatch')
 tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 selected={lang:[] for lang in LANGS};reasons=collections.Counter();dedup=set()
 for raw in gzip.open(src,'rt',encoding='utf-8'):
  x=json.loads(raw);lang=x.get('lang')
  if lang not in LANGS or x.get('role')!='prompter' or x.get('parent_id') is not None or x.get('deleted') or x.get('synthetic') or not x.get('review_result'):
   reasons['not_eligible']+=1;continue
  labels=x.get('labels') or {}
  if any(float((labels.get(k) or {}).get('value') or 0)>.000001 for k in
      ('spam','pii','hate_speech','sexual_content','violence','not_appropriate','toxicity')):
   reasons['flagged_public_user_content']+=1;continue
  text=(x.get('text') or '').strip()
  if BAD.search(text):reasons['potential_personal_identifier_or_link']+=1;continue
  if not (25<=len(text)<=700):reasons['char_length']+=1;continue
  if sum(text.count(s) for s in ('<script','-----BEGIN','file://'))>0:
   reasons['unexpected_markup']+=1;continue
  h=hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()
  if h in dedup:reasons['duplicate']+=1;continue
  dedup.add(h)
  num=len(tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
  if num>880 or num<9:reasons['token_length']+=1;continue
  selected[lang].append({'request_id':'oasst-'+h[:20],'source_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
    'lang':lang,'source_prompt':text,'prompt_tokens_no_system':num})
 final=[]
 for lang in LANGS:
  rows=sorted(selected[lang],key=lambda x:hashlib.sha256(('oai-stage35-external-20261030'+x['source_text_sha256']).encode()).hexdigest())
  if len(rows)<32:raise RuntimeError('insufficient safe human prompts '+lang+': '+str(len(rows)))
  final.extend(rows[:32])
 final.sort(key=lambda x:hashlib.sha256(('stage35-global-order'+x['source_text_sha256']).encode()).hexdigest())
 if len(final)!=96 or len({r['source_text_sha256'] for r in final})!=96:raise ValueError('not 96 unique identities')
 private_dir.mkdir(parents=True,exist_ok=True)
 payload=''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n' for row in final).encode()
 (private_dir/'stage35_selected_user_prompts.jsonl').write_bytes(payload)
 (private_dir/'stage35_selected_user_prompts.jsonl').chmod(0o600)
 systemhash=hashlib.sha256(SYSTEM.encode()).hexdigest()
 (private_dir/'stage35_shared_system_prompt.txt').write_text(SYSTEM)
 records=[{'request_id':r['request_id'],'lang':r['lang'],'content_sha256':r['source_text_sha256'],
    'prompt_tokens_no_system':r['prompt_tokens_no_system']} for r in final]
 pub={'schema':'stage35-public-oasst1-human-root-v1','source':'OpenAssistant/oasst1 2023-04-12_oasst_prompts.messages.jsonl.gz',
  'source_url':'https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_prompts.messages.jsonl.gz',
  'source_sha256':SOURCE,'license':'Apache-2.0, OpenAssistant LAION AI',
  'sampling':'32 original roots en/es/zh, approved human reviewed, no synthetic/deleted, zero flagged PII/toxicity/NSFW/spam, strip identifiers and URLs, 25..700 chars, <=880 Qwen tokens, SHA256 order',
  'request_count':96,'lang_counts':dict(collections.Counter(x['lang'] for x in final)),
  'private_selected_jsonl_sha256':hashlib.sha256(payload).hexdigest(),
  'private_selected_local_path':'results/stage35_public_data/stage35_selected_user_prompts.jsonl',
  'raw_user_prompts_not_committed':True,'public_request_hashes':records,
  'system_prompt_sha256':systemhash,'system_prompt_tokens':len(tokenizer.encode(SYSTEM)),
  'frozen_seeds':[3511,3527,3559], 'conditions':{'system_shared_fraction':[0,.5,1], 'cache':['off','on']},
  'filter_counts':dict(reasons),
  'limitations':['Prompts are public crowd-sourced user data but may contain undetected personal information; raw prompt texts remain on Tang, not GitHub',
   'Sampling and forced language proportions are not representative of all production traffic',
   'Model may have encountered public OASST1 during pretraining']}
 public_dir.mkdir(parents=True,exist_ok=True)
 (public_dir/'stage35_oasst_manifest.json').write_text(json.dumps(pub,indent=2,ensure_ascii=False)+'\n')
 print('STAGE35_PUBLIC_MANIFEST_FROZEN',pub['request_count'],pub['lang_counts'],pub['private_selected_jsonl_sha256'])
 print('FILTERED',dict(reasons),'system_token_count',pub['system_prompt_tokens'])
if __name__=='__main__':
 build(Path('results/stage35_public_data/oasst1_prompts.messages.jsonl.gz'),Path('results/stage35_public_data'),Path('data'))
