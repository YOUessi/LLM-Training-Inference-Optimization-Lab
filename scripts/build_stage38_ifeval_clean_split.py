"""Frozen, identity-disjoint instruction-dev and untouched holdout from official IFEval 541.

Stage36's 96 examined examples are entirely excluded to avoid adaptive reuse.
No model scores or generated answers are accessed during split construction.
"""
from __future__ import annotations
import collections,hashlib,json
from pathlib import Path
from build_stage36_ifeval import eligible, SRC_SHA
DEV_PER_CLASS=5;TEST_PER_CLASS=12
CLASS_OVERRIDE={'language':(3,5)}
SALT='stage38-guarded-adapter-strength-2026-10-30'

def build(src:Path,data:Path):
 from transformers import AutoTokenizer
 if hashlib.sha256(src.read_bytes()).hexdigest()!=SRC_SHA:raise ValueError('official source changed')
 tokenizer=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 old=[json.loads(s) for s in (data/'stage36_ifeval_96.jsonl').read_text().splitlines()]
 oldkeys={x['key'] for x in old}
 if len(oldkeys)!=96:raise ValueError('Stage36 old evaluated identities changed')
 groups=collections.defaultdict(list)
 for line in src.read_text().splitlines():
  r=json.loads(line)
  if r['key'] in oldkeys:continue
  ok,reason=eligible(r)
  if not ok:continue
  prompt_token_count=len(tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=True,
     add_generation_prompt=True,enable_thinking=False))
  if prompt_token_count+384>2048:continue
  family=r['instruction_id_list'][0].split(':')[0]
  groups[family].append(r)
 expected={'change_case','combination','detectable_content','detectable_format','keywords','language','length_constraints','punctuation','startend'}
 if set(groups)!=expected:raise ValueError('missing original instruction families')
 dev=[];test=[];counts={}
 for family in sorted(groups):
  count_dev,count_test=CLASS_OVERRIDE.get(family,(DEV_PER_CLASS,TEST_PER_CLASS))
  options=sorted(groups[family],key=lambda row:hashlib.sha256((SALT+':'+str(row['key'])).encode()).hexdigest())
  if len(options)<count_dev+count_test:raise ValueError('insufficient unseen examples '+family)
  dev.extend(options[:count_dev]);test.extend(options[count_dev:count_dev+count_test])
  counts[family]={'dev':count_dev,'test':count_test,'unused_eligible':len(options)-count_dev-count_test}
 for split,rows in [('dev',dev),('test',test)]:
  rows.sort(key=lambda row:hashlib.sha256((SALT+':'+split+':'+str(row['key'])).encode()).hexdigest())
 if len(dev)!=43 or len(test)!=101:raise ValueError('unexpected dev/test count')
 devkeys={x['key'] for x in dev};testkeys={x['key'] for x in test}
 if len(devkeys)!=43 or len(testkeys)!=101 or devkeys&testkeys or devkeys&oldkeys or testkeys&oldkeys:
  raise ValueError('stage36/dev/final test key overlap')
 manifests={}
 for split,rows in (('dev',dev),('test',test)):
  payload=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rows).encode()
  name=f'stage38_ifeval_{split}.jsonl';(data/name).write_bytes(payload)
  manifests[split]={'count':len(rows),'sha256':hashlib.sha256(payload).hexdigest(),
   'keys':[x['key'] for x in rows],'category_counts':dict(collections.Counter(x['instruction_id_list'][0].split(':')[0] for x in rows))}
 manifest={'schema':'stage38-clean-disjoint-ifeval-dev-test-v1','source_name':'google/IFEval',
  'original_source_sha256':SRC_SHA,'license':'Apache-2.0 Google Research IFEval',
  'excluded_stage36_keys_count':len(oldkeys),
  'stage36_keys_sha256':hashlib.sha256(json.dumps(sorted(oldkeys)).encode()).hexdigest(),
  'sampling':'Fixed per-tier counts with predeclared language override, deterministic SHA256 before all new generation; Stage36 96 keys excluded',
  'class_assignments':counts,'splits':manifests,'test_must_remain_untouched_until_scale_selected':True,
  'max_generation_tokens':384,'source_code':'scripts/build_stage38_ifeval_clean_split.py',
  'limitations':['Both dev and test are from same public IFEval source, not independent user traffic',
    'Existing Stage36 96 are excluded because already observed and used to motivate experiment',
    'Test 101 is untouched by scale selection; model may have seen public prompts in pretraining',
    'Output cap384 may prevent some prompt constraints from being met']}
 (data/'stage38_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('STAGE38_FROZEN_CLEAN_IFEVAL',len(dev),'DEV',len(test),'UNSEEN_TEST',manifests['test']['sha256'])
 print('CLASS_COUNTS',counts)
 return manifest
if __name__=='__main__':build(Path('results/stage35_public_data/ifeval_input_data.jsonl'),Path('data'))
