"""Untouched Dolly128 CE for pre-frozen base, original SFT, scaled LoRA & format replay.

All model/strength selections were made earlier on **different validation sets**;
Stage41 test never controls training, scaling or candidate selection.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,random,statistics,time
from pathlib import Path
from llm_opt_lab.training.stage26_real_sft import load_split,collate,MODEL

def audit_sources(data:Path):
 manifest=json.loads((data/'stage41_manifest.json').read_text())
 blob=(data/'stage41_dolly_holdout128.jsonl').read_bytes()
 if hashlib.sha256(blob).hexdigest()!=manifest['frozen_input_sha256']:
  raise ValueError('Stage41 new source checksum changed')
 old=set()
 for split in ('train','validation','test'):
  old.update(json.loads(x)['identity_sha256'] for x in (data/f'stage26_{split}.jsonl').read_text().splitlines())
 old.update(json.loads(x)['identity_sha256'] for x in (data/'stage34_new_holdout.jsonl').read_text().splitlines())
 new={json.loads(x)['identity_sha256'] for x in blob.decode().splitlines()}
 if len(old)!=832 or len(new)!=128 or old&new:raise ValueError('Stage41 test identities leaked')
 return manifest

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',type=Path,default=Path('data'))
 p.add_argument('--stage38',type=Path,required=True)
 p.add_argument('--stage40',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 args=p.parse_args()
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from peft import PeftModel
 from llm_opt_lab.training.stage38_adapter_strength import set_scale_factor
 meta=audit_sources(args.data)
 decision38=json.loads((args.stage38/'frozen_dev_selection.json').read_text())
 decision40=json.loads((args.stage40/'dev_selection.json').read_text())
 if decision38['status']!='dev_approved_for_holdout_not_production' or not decision38['holdout_not_loaded_or_used_for_selection']:
  raise ValueError('Stage38 strength not frozen on dev')
 if decision40['status']!='dev_frozen_no_final_test_access' or not decision40['final_test_unaccessed']:
  raise ValueError('Stage40 training candidate not frozen by dev')
 original=Path('evidence/stage31_full512/selected_adapter')
 alt=None
 if decision40['selected']!='base':
  alt=args.stage40/(decision40['selected']+'_adapter')
  if not alt.is_dir():raise FileNotFoundError('Stage40 dev selected adapter not found')
 t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 test=load_split(args.data/'stage41_dolly_holdout128.jsonl',t)
 if len(test)!=128 or len({r['identity_sha256'] for r in test})!=128:raise ValueError('missing untouched holdout')
 torch.manual_seed(410041);torch.cuda.manual_seed_all(410041)
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,device_map={'':0},
  dtype=torch.bfloat16,quantization_config=q,attn_implementation='sdpa')
 base.config.use_cache=False
 model=PeftModel.from_pretrained(base,original,is_trainable=False,adapter_name='old_full')
 if alt:model.load_adapter(alt,adapter_name='format_replay',is_trainable=False)
 model.eval();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
 from contextlib import nullcontext
 alphas={'base':0.,'original_full':1.,'guarded_strength':float(decision38['selected_alpha'])}
 results={}
 for label in ('base','original_full','guarded_strength','format_replay'):
  if label=='base' or (label=='guarded_strength' and alphas['guarded_strength']==0.) or (label=='format_replay' and alt is None):
   ctx=model.disable_adapter()
  elif label=='guarded_strength':
   model.set_adapter('old_full')
   ctx=set_scale_factor(model,alphas['guarded_strength'],adapter_name='old_full')
  else:
   model.set_adapter('old_full' if label=='original_full' else 'format_replay')
   ctx=nullcontext()
  entries=[]
  with ctx,torch.inference_mode():
   for i,row in enumerate(test):
    batch=collate([row],t.eos_token_id)
    loss=float(model(**batch,use_cache=False).loss.detach())
    if not math.isfinite(loss):raise ValueError('bad Stage41 assistant CE')
    entries.append({'identity_sha256':row['identity_sha256'],'category':row['category'],
     'answer_tokens':row['answer_count'],'ce_nats':loss})
    if (i+1)%32==0:print('STAGE41_FRESH_CE',label,i+1,'of128',flush=True)
  results[label]=entries
  print('STAGE41_MODEL',label,'CE',round(sum(x['ce_nats']*x['answer_tokens'] for x in entries)/sum(x['answer_tokens'] for x in entries),5),flush=True)
 categories=sorted({x['category'] for x in test})
 def weighted(items):
  n=sum(x['answer_tokens'] for x in items)
  ce=sum(x['ce_nats']*x['answer_tokens'] for x in items)/n
  return {'examples':len(items),'answer_tokens':n,'assistant_ce_nats':ce,'perplexity':math.exp(ce)}
 aggregate={label:weighted(records) for label,records in results.items()}
 by_category={category:{label:weighted([x for x in records if x['category']==category]) for label,records in results.items()}
  for category in categories}
 randomizer=random.Random(410041);samples={c:[r['identity_sha256'] for r in test if r['category']==c] for c in categories}
 lookup={label:{r['identity_sha256']:r for r in rows} for label,rows in results.items()}
 cis={}
 for altlabel in ('original_full','guarded_strength','format_replay'):
  deltas=[]
  for _ in range(3000):
   chosen=[refs[randomizer.randrange(16)] for refs in samples.values() for _ in range(16)]
   total=sum(lookup[altlabel][key]['answer_tokens'] for key in chosen)
   val=sum((lookup[altlabel][key]['ce_nats']-lookup['base'][key]['ce_nats'])*lookup[altlabel][key]['answer_tokens'] for key in chosen)/total
   deltas.append(val)
  deltas.sort();cis[altlabel]=[deltas[75],deltas[2924]]
 output={'status':'measured','kind':'stage41_128_independent_ID_dolly_answerCE_multiobjective_adapter_comparison',
  'source_sha256':meta['source_sha256'],'holdout_sha256':meta['frozen_input_sha256'],
  'excluded_previous832_identites':True,
  'selection38_dev_alpha':float(decision38['selected_alpha']),
  'selection40_dev_model':decision40['selected'],
  'selection38_file_sha256':hashlib.sha256((args.stage38/'frozen_dev_selection.json').read_bytes()).hexdigest(),
  'selection40_file_sha256':hashlib.sha256((args.stage40/'dev_selection.json').read_bytes()).hexdigest(),
  'adapters':{'original_stage31':hashlib.sha256((original/'adapter_model.safetensors').read_bytes()).hexdigest(),
   'stage40_formatted':hashlib.sha256((alt/'adapter_model.safetensors').read_bytes()).hexdigest() if alt else None},
  'counts':{'examples':128,'categories':8,'models':4},'overall':aggregate,'by_category':by_category,
  'paired_stratified_bootstrap_delta_ce_vs_original_base':cis,
  'per_example_answer_ce':results,'evaluation_wall_s':time.perf_counter()-started,
  'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
  'limits':['128 new *record identities* versus prior832, same public Dolly dataset/model pretraining leakage may remain',
   'No model or scale selected on this test; results do not prove semantic correctness/human preference',
   'Four model conditions may share same actual adapter when guarded selection defaults base',
   'Only single trained seed selected via IFEval/Dolly dev, not an independent rank/seed ensemble']}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(output,indent=2)+'\n')
 print('STAGE41_NEW128_TRIPLE_GUARD_CE_VERIFIED',aggregate,flush=True)
if __name__=='__main__':main()
