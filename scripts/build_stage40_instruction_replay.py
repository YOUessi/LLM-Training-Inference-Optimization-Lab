"""Deterministic, non-IFEval constraint-format Dolly rehearsal + clean IFEval dev/test.

Format targets are deterministic reshaping of *human* Dolly answers, not model
hallucinations, and every selected assistant output is complete <=384 tokens.
New IFEval dev/test exclude all Stage36/38 previously evaluated identities.
"""
from __future__ import annotations
import hashlib,json,collections,html
from pathlib import Path
from build_stage36_ifeval import SRC_SHA,eligible
SEED_SALT='stage40-constraint-replay-20261030'
FAMILIES=('change_case','combination','detectable_format','keywords','startend')
DEV_PER_FAMILY=5
TEST_PER_FAMILY=12
DERIVED_VARIANTS=('json_exact','xml_tag','begin_end')
MAX_LEN=384

def formatted(example,kind):
 user=example['instruction']+('\n\nReference context:\n'+example['context'] if example['context'] else '')
 answer=example['response']
 if kind=='json_exact':
  return (user+'\n\nReturn your response as one valid JSON object with exactly the key "answer" whose value is a string. '+
    'No code fences and no commentary outside the JSON.',json.dumps({'answer':answer},ensure_ascii=False,separators=(',',':')))
 if kind=='xml_tag':
  return (user+'\n\nPlace the entire answer inside exactly one <answer>...</answer> tag pair. '+
    'Do not add any text before or after the tag.', '<answer>'+html.escape(answer,quote=True)+'</answer>')
 if kind=='begin_end':
  return (user+'\n\nBegin the answer with START_ANSWER on its own line, and end with END_ANSWER on its own line. '+
    'Everything else must be between these exact markers.', 'START_ANSWER\n'+answer+'\nEND_ANSWER')
 raise ValueError('unregistered transform '+kind)

def build(data:Path,ifeval_src:Path):
 from transformers import AutoTokenizer
 t=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 original=json.loads((data/'stage26_manifest.json').read_text())
 raw=(data/'stage26_train.jsonl').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=original['splits']['train']['sha256']:raise ValueError('Dolly training source mutated')
 rows=[json.loads(x) for x in raw.decode().splitlines()]
 if len(rows)!=512 or len({r['identity_sha256'] for r in rows})!=512:raise ValueError('not exact original 512 Dolly identities')
 augmented=[];eligibility=collections.Counter()
 for record in rows:
  variants={}
  for kind in DERIVED_VARIANTS:
   prompt,target=formatted(record,kind)
   a=t.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
   b=t.apply_chat_template([{'role':'user','content':prompt},{'role':'assistant','content':target}],tokenize=True,
      add_generation_prompt=False,enable_thinking=False)
   if len(b)>MAX_LEN or b[:len(a)]!=a or len(b)-len(a)<2:
    eligibility[kind+'_cannot_fit']+=1;continue
   if kind=='json_exact' and json.loads(target)!={'answer':record['response']}:
    raise ValueError('bad JSON reference label')
   if kind=='xml_tag' and not(target.startswith('<answer>') and target.endswith('</answer>')):
    raise ValueError('bad XML label')
   if kind=='begin_end' and not(target.startswith('START_ANSWER\n') and target.endswith('\nEND_ANSWER')):
    raise ValueError('bad marker label')
   variants[kind]={'user':prompt,'answer':target,'input_tokens':len(a),'total_tokens':len(b)}
   eligibility[kind+'_valid']+=1
  augmented.append({'identity_sha256':record['identity_sha256'],'category':record['category'],
    'original_instruction':record['instruction'],'original_context':record['context'],
    'original_response':record['response'],'derived':variants})
 if len(augmented)!=512:raise ValueError('missing source rows')
 payload=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in augmented).encode()
 (data/'stage40_dolly_format_replay.jsonl').write_bytes(payload)
 if hashlib.sha256(ifeval_src.read_bytes()).hexdigest()!=SRC_SHA:raise ValueError('official IFEval 541 SHA mismatch')
 seen=set()
 for file in ('stage36_ifeval_96.jsonl','stage38_ifeval_dev.jsonl','stage38_ifeval_test.jsonl'):
  seen.update(json.loads(s)['key'] for s in (data/file).read_text().splitlines())
 if len(seen)!=240:raise ValueError('IFEval already-examined identity count drift')
 groups=collections.defaultdict(list)
 for line in ifeval_src.read_text().splitlines():
  r=json.loads(line)
  if r['key'] in seen or not eligible(r)[0]:continue
  family=r['instruction_id_list'][0].split(':')[0]
  if family not in FAMILIES:continue
  a=t.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  if len(a)+384>2048:continue
  groups[family].append(r)
 dev=[];test=[]
 for tier in FAMILIES:
  options=sorted(groups[tier],key=lambda r:hashlib.sha256((SEED_SALT+str(r['key'])).encode()).hexdigest())
  if len(options)<DEV_PER_FAMILY+TEST_PER_FAMILY:raise ValueError('tier insufficient '+tier)
  dev.extend(options[:DEV_PER_FAMILY]);test.extend(options[DEV_PER_FAMILY:DEV_PER_FAMILY+TEST_PER_FAMILY])
 dev.sort(key=lambda x:hashlib.sha256((SEED_SALT+'dev'+str(x['key'])).encode()).hexdigest())
 test.sort(key=lambda x:hashlib.sha256((SEED_SALT+'test'+str(x['key'])).encode()).hexdigest())
 if len(dev)!=25 or len(test)!=60:raise ValueError('bad 25/60 dev/test')
 split_manifest={}
 for split,records in (('dev',dev),('test',test)):
  data_bytes=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in records).encode()
  (data/f'stage40_ifeval_{split}.jsonl').write_bytes(data_bytes)
  split_manifest[split]={'records':len(records),'sha256':hashlib.sha256(data_bytes).hexdigest(),
   'keys':[r['key'] for r in records],
   'family_counts':dict(collections.Counter(r['instruction_id_list'][0].split(':')[0] for r in records))}
 allseen=set(split_manifest['dev']['keys'])|set(split_manifest['test']['keys'])
 if len(allseen)!=85 or allseen&seen:raise ValueError('Stage40 IFEval leakage')
 doc={'status':'frozen','source_dolly_manifest_sha256':hashlib.sha256((data/'stage26_manifest.json').read_bytes()).hexdigest(),
  'original_dolly_train_sha256':original['splits']['train']['sha256'],
  'dolly_format_replay_sha256':hashlib.sha256(payload).hexdigest(),
  'format_families':list(DERIVED_VARIANTS),'max_full_chat_tokens':MAX_LEN,
  'training_source_license':'Databricks Dolly 15k CC BY-SA 3.0; synthetic deterministic formatting transformations are derivative under same license',
  'format_variant_valid_counts':dict(eligibility),
  'ifeval_original_src_sha256':SRC_SHA,
  'ifeval_dev_test':split_manifest,'prior_seen_ifeval_keys_count':len(seen),
  'ifeval_source_license':'Apache-2.0 Google/IFEval',
  'ifeval_selected_families':list(FAMILIES),
  'no_eval_prompt_used_as_training_target':True,
  'limitations':['Constraint demonstrations are synthetic deterministic rewrites of Dolly human reference answers, not independently human judged',
   'Only five instruction categories have sufficient strictly untouched IFEval prompts after stages36/38',
   'Source models may have seen public Dolly/IFEval during pretraining',
   'Format-constrained labels can still induce lexical/style overfitting']}
 (data/'stage40_manifest.json').write_text(json.dumps(doc,indent=2)+'\n')
 print('STAGE40_DATA_FROZEN',len(augmented),'DOLLY',len(dev),'DEV',len(test),'UNTOUCHED_TEST',dict(eligibility))
 return doc
if __name__=='__main__':build(Path('data'),Path('results/stage35_public_data/ifeval_input_data.jsonl'))
