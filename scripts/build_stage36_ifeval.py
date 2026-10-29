"""Freeze 96 balanced feasible official IFEval instructions before any generation.

Rows are selected by stable SHA ranking and computability of length requirements
under 384 decoded-token cap, NOT by observed model behavior.
"""
import hashlib,json,collections
from pathlib import Path
SRC_SHA='6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2'
UPSTREAM_COMMIT='7e672ccbb111603f237a3daed2f98dccd9300023'
CAP=384

def eligible(r):
 from transformers import AutoTokenizer
 # fixed resource/correctness filters, no model outputs consulted.
 for inst,kw in zip(r['instruction_id_list'],r['kwargs']):
  if inst=='length_constraints:number_sentences':return False,'requires_punkt_corpus_not_preinstalled'
  if inst=='length_constraints:number_words':
   if kw.get('relation')=='at least' and int(kw.get('num_words',0))>175:
    return False,'requested_min_words_exceeds_preregistered_384_token_budget'
  if inst=='combination:repeat_prompt' and len(r['prompt'])>1200:
   return False,'very_long_repeat_prompt'
 return True,None

def build(source:Path,out:Path):
 from transformers import AutoTokenizer
 raw=source.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=SRC_SHA:raise ValueError('upstream IFEval SHA changed')
 tok=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 rows=[json.loads(line) for line in raw.decode().splitlines()]
 if len(rows)!=541:raise ValueError('expected official 541 input rows')
 groups={};filtered=collections.Counter();all_tokens={}
 for r in rows:
  if len(set(r))!=4 or not set(r)=={'key','prompt','instruction_id_list','kwargs'}:raise ValueError('original IFEval keys changed')
  if len(r['instruction_id_list'])!=len(r['kwargs']):raise ValueError('kwargs list mismatch')
  ok,reason=eligible(r)
  if not ok:filtered[reason]+=1;continue
  tokens=len(tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,
       add_generation_prompt=True,enable_thinking=False))
  if tokens+CAP>2048:filtered['input_plus_max_gen_context_too_big']+=1;continue
  category=r['instruction_id_list'][0].split(':')[0]
  groups.setdefault(category,[]).append(r)
  all_tokens[r['key']]=tokens
 print('IFEVAL_CATEGORIES_AVAILABLE',[(k,len(v)) for k,v in groups.items()])
 if len(groups)<8:raise ValueError('not enough instruction coverage')
 # Exactly 96 balanced across first-level instruction families; simple deterministic quota allocation.
 categories=sorted(groups)
 quota=96//len(categories)
 leftover=96%len(categories)
 final=[]
 for i,category in enumerate(categories):
  order=sorted(groups[category],key=lambda r:hashlib.sha256(('stage36-ifeval-20261030'+str(r['key'])).encode()).hexdigest())
  n=quota+(1 if i<leftover else 0)
  if len(order)<n:raise ValueError('insufficient category '+category+' '+str(len(order)))
  final+=order[:n]
 final=sorted(final,key=lambda r:hashlib.sha256(('stage36-fixed-display-order'+str(r['key'])).encode()).hexdigest())
 if len(final)!=96 or len({r['key'] for r in final})!=96:raise ValueError('incorrect final selection')
 out.mkdir(parents=True,exist_ok=True)
 data=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in final).encode()
 (out/'stage36_ifeval_96.jsonl').write_bytes(data)
 record={'schema':'stage36-official-IFEval-96-v1', 'license':'Apache-2.0 (Google IFEval)',
  'source':'google/IFEval 541 benchmark instruction records','source_original_sha256':SRC_SHA,
  'source_download':'https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl',
  'official_evaluator':'google-research/google-research/instruction_following_eval',
  'evaluator_source_commit':UPSTREAM_COMMIT,
  'selection':'Deterministic SHA-sorted categories on original official instructions, equal-ish quotas per tier-0 family; no model results consulted',
  'filter_rules':{'exclude_number_sentences':'official evaluation requires unavailable NLTK Punkt training data',
    'exclude_min_words_gt175':'384-token cap makes longer word minimum infeasible',
    'exclude_repetition_prompt_over_1200chars':'repeat long prompt incompatible with output cap',
    'require_input_tokens_plus_output_cap_le2048':True},
  'count':96,'category_counts':dict(collections.Counter(r['instruction_id_list'][0].split(':')[0] for r in final)),
  'selected_source_keys':[r['key'] for r in final],
  'max_generated_tokens':CAP,'frozen_input_sha256':hashlib.sha256(data).hexdigest(),
  'prompt_tokens':{str(r['key']):all_tokens[r['key']] for r in final},
  'excluded_reasons':dict(filtered),'notes':['Subset of official 541; not the official full-dataset IFEval leaderboard score',
   'Selection before evaluating models, no contamination of Stage26/31 training by IFEval source',
   'Strict and loose rule-based evaluator measures verifiable constraint obedience, not factual accuracy']}
 (out/'stage36_ifeval_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
 print('STAGE36_IFEVAL96_FROZEN',len(final),record['frozen_input_sha256'],record['category_counts'],record['excluded_reasons'])
if __name__=='__main__':build(Path('results/stage35_public_data/ifeval_input_data.jsonl'),Path('data'))
