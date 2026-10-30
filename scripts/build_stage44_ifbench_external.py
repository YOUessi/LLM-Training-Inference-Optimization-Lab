"""Freeze external-source IFBench60 OOD instruction benchmark before model results.

Original WildChat-derived prompts may contain personal material, so raw text is
kept only locally; public manifest contains hashes and keys, no raw prompts.
"""
from __future__ import annotations
import hashlib,json,collections
from pathlib import Path
SOURCE_SHA='a2bf7b8e7ed11d39c65b4ec781da8b99663ed0121f97033d70fcecb61f8a9d85'
COMMIT='1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d'
SALT='stage44-ifbench-ood-test-preserve-first-look-20261030'
QUOTAS={'format':14,'words':14,'count':10,'ratio':8,'sentence':7,'custom':4,'repeat':3}

def build(source:Path,public:Path,private:Path):
 from transformers import AutoTokenizer
 if hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA:raise ValueError('IFBench official source SHA changed')
 data=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
 if len(data)!=300 or sum(QUOTAS.values())!=60:raise ValueError('unexpected official IFBench test size')
 tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 groups=collections.defaultdict(list);reject=collections.Counter()
 for row in data:
  if set(row)!={'key','prompt','instruction_id_list','kwargs'}:raise ValueError('unexpected IFBench original schema')
  if not row['instruction_id_list'] or len(row['instruction_id_list'])!=len(row['kwargs']):
   raise ValueError('unsupported IFBench instruction structure')
  category=row['instruction_id_list'][0].split(':')[0]
  if category not in QUOTAS:raise ValueError('new IFBench family unregistered '+category)
  tokens=len(tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,
         add_generation_prompt=True,enable_thinking=False))
  if tokens+384>2048:
   reject['context_overflow']+=1;continue
  if not row['prompt'].strip():reject['empty_prompt']+=1;continue
  groups[category].append(row)
 selected=[]
 for family,n in QUOTAS.items():
  order=sorted(groups[family],key=lambda r:hashlib.sha256((SALT+str(r['key'])).encode()).hexdigest())
  if len(order)<n:raise ValueError(f'insufficient eligible IFBench {family}: {len(order)} < {n}')
  selected+=order[:n]
 selected.sort(key=lambda r:hashlib.sha256((SALT+'order'+str(r['key'])).encode()).hexdigest())
 if len(selected)!=60 or len({str(r['key']) for r in selected})!=60:raise ValueError('not exactly 60 unique rows')
 content=''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n' for row in selected).encode()
 private.mkdir(parents=True,exist_ok=True)
 dest=private/'stage44_ifbench_external60.jsonl';dest.write_bytes(content);dest.chmod(0o600)
 manifest={'schema':'stage44-external-IFBench60-ood-v1','source_repo':'allenai/IFBench',
  'source_commit':COMMIT,'source_original_sha256':SOURCE_SHA,
  'source_url':'https://github.com/allenai/IFBench/blob/main/ifbench/data/IFBench_test.jsonl',
  'dataset_license':'ODC-BY-1.0, with responsible-use guidelines and third-party generated data terms',
  'code_license':'Apache-2.0, original official IFBench verifier',
  'sample_count':60,'quota_by_family':QUOTAS,
  'selected_sha256':hashlib.sha256(content).hexdigest(),
  'source_keys_and_prompt_hashes':[{'key':str(r['key']),'prompt_sha256':hashlib.sha256(r['prompt'].encode()).hexdigest(),
     'constraint_ids':r['instruction_id_list']} for r in selected],
  'private_selected_file':'results/stage44_ifbench_source/stage44_ifbench_external60.jsonl',
  'private_raw_text_not_committed':True,
  'preregistered_model_comparison':['base','lora_sft','stage42_prompt_only_router'],
  'generation':{'max_new_tokens':384,'thinking_disabled':True,'greedy':True},
  'no_model_outputs_used_for_selection':True,
  'source_filter_exclusions':dict(reject),
  'limitations':['New IFBench constraint families beyond classic IFEval, but 60-case stratified subset of 300, not official leaderboard',
    'The original IFBench test prompts come partly from public WildChat and may contain personal data; raw text kept locally',
    'Model may have been exposed to public IFBench source during pretraining',
    'IFBench benchmark has some target constraints requiring longer than 384 tokens; report any cap-induced failure and do not retune post hoc',
    'Official IFBench verifier is a constraint-checker, not semantic/human correctness or safety reviewer']}
 public.mkdir(parents=True,exist_ok=True)
 (public/'stage44_ifbench_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 print('STAGE44_OUT_OF_DOMAIN_IFBENCH60_FROZEN',manifest['selected_sha256'],manifest['quota_by_family'],dict(reject))
 return manifest
if __name__=='__main__':
 build(Path('results/stage44_ifbench_source/IFBench_test.jsonl'),Path('data'),Path('results/stage44_ifbench_source'))
