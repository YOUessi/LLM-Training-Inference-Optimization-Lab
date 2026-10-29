"""Truly unused source IDs: 128 Dolly examples vs base, 192-exposure SFT and 1024-exposure SFT.

Dataset source is unchanged Dolly 15k, but all selected 128 records were excluded
from Stage26's train/validation/test 704 IDs before the 8B training outcomes.
No test feedback used for model selection; this is an independent-record CE check.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics,time
from pathlib import Path

def audit_data(root:Path):
 manifest=json.loads((root/'stage34_manifest.json').read_text())
 data=(root/'stage34_new_holdout.jsonl').read_bytes()
 if hashlib.sha256(data).hexdigest()!=manifest['input_sha256']:raise ValueError('Stage34 frozen data drift')
 if len(data.splitlines())!=128 or manifest['record_count']!=128:raise ValueError('not 128 holdout records')
 previous=set()
 for split in ('train','validation','test'):
  previous.update(json.loads(l)['identity_sha256'] for l in (root/f'stage26_{split}.jsonl').read_text().splitlines() if l.strip())
 new=[json.loads(l)['identity_sha256'] for l in data.decode().splitlines()]
 if len(previous)!=704 or len(set(new))!=128 or any(i in previous for i in new):raise ValueError('test leakage')
 return manifest

def summarize(rows):
 toks=sum(x['answer_tokens'] for x in rows)
 ce=sum(x['ce_nats']*x['answer_tokens'] for x in rows)/toks
 return {'records':len(rows),'assistant_tokens':toks,'ce_nats':ce,'perplexity':math.exp(ce),
  'mean_record_ce':statistics.mean(x['ce_nats'] for x in rows)}

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--input',type=Path,default=Path('data'))
 ap.add_argument('--stage31',type=Path,required=True)
 ap.add_argument('--stage26-adapter',type=Path,default=Path('evidence/stage26_sft_v2/selected_B_adapter'))
 ap.add_argument('--out',type=Path,required=True)
 a=ap.parse_args()
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import PeftModel
 from llm_opt_lab.training.stage26_real_sft import load_split,collate
 manifest=audit_data(a.input)
 meta=json.loads((a.stage31/'selection.json').read_text())
 if meta['status']!='candidate_selected_from_validation_only' or not meta['test_has_not_been_accessed']:
  raise ValueError('Stage31 selection provenance invalid')
 chosen=a.stage31/(meta['selected']+'_adapter')
 if not chosen.is_dir() or not a.stage26_adapter.is_dir():raise ValueError('both frozen selected adapters required')
 tok=AutoTokenizer.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True)
 rows=load_split(a.input/'stage34_new_holdout.jsonl',tok)
 if len(rows)!=128 or len({x['identity_sha256'] for x in rows})!=128:raise ValueError('invalid Stage34 tokenization')
 torch.manual_seed(340034);torch.cuda.manual_seed_all(340034)
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained('/home/you/models/Qwen3-8B',local_files_only=True,
  device_map={'':0},dtype=torch.bfloat16,quantization_config=q,attn_implementation='sdpa')
 base.config.use_cache=False
 model=PeftModel.from_pretrained(base,a.stage26_adapter,adapter_name='stage26',is_trainable=False)
 model.load_adapter(chosen,adapter_name='stage31',is_trainable=False)
 model.eval()
 table={}
 torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 with torch.inference_mode():
  for variant in ('base','stage26_short','stage31_full'):
   if variant=='base':ctx=model.disable_adapter()
   else:
    model.set_adapter('stage26' if variant=='stage26_short' else 'stage31')
    from contextlib import nullcontext
    ctx=nullcontext()
   subset=[]
   with ctx:
    for i,row in enumerate(rows):
     batch=collate([row],tok.eos_token_id)
     ce=float(model(**batch,use_cache=False).loss.detach())
     if not math.isfinite(ce):raise ValueError('nonfinite test CE')
     subset.append({'identity_sha256':row['identity_sha256'],'category':row['category'],
       'answer_tokens':row['answer_count'],'ce_nats':ce})
     if (i+1)%32==0:print('STAGE34_CE_PROGRESS',variant,i+1,'of128',flush=True)
   table[variant]=subset
 categories=sorted({r['category'] for r in rows})
 overview={m:summarize(rs) for m,rs in table.items()}
 category_scores={c:{m:summarize([r for r in rs if r['category']==c]) for m,rs in table.items()} for c in categories}
 for c in categories:
  if any(category_scores[c][m]['records']!=16 for m in table):raise ValueError('16 per source category required')
 identities=[r['identity_sha256'] for r in rows]
 for m,rs in table.items():
  if [x['identity_sha256'] for x in rs]!=identities:raise ValueError('unpaired per-record order')
 rng=random.Random(3400034)
 uncertainties={}
 for comparator in ('base','stage26_short'):
  original={m:{r['identity_sha256']:r for r in rs} for m,rs in table.items()}
  groups={c:[r['identity_sha256'] for r in rows if r['category']==c] for c in categories}
  values=[]
  for _ in range(2000):
   sample=[]
   for c in categories:
    sample.extend(groups[c][rng.randrange(16)] for _ in range(16))
   total=sum(original['stage31_full'][ident]['answer_tokens'] for ident in sample)
   new=sum(original['stage31_full'][ident]['ce_nats']*original['stage31_full'][ident]['answer_tokens'] for ident in sample)/total
   old=sum(original[comparator][ident]['ce_nats']*original[comparator][ident]['answer_tokens'] for ident in sample)/total
   values.append(new-old)
  values.sort()
  uncertainties[comparator]={'delta_new_minus_comparator_ce':overview['stage31_full']['ce_nats']-overview[comparator]['ce_nats'],
   'within_fixed_128_stratified_bootstrap_95pct_ci':[values[50],values[1949]]}
 result={'status':'measured','kind':'stage34_128_fresh_unused_dolly8b_qlora_cross_generalization_CE',
  'source_sha256':manifest['source_original_sha256'],'frozen_unused_holdout_sha256':manifest['input_sha256'],
  'source_ids_excluded_from_all_previous_704':True,'selected_stage31':meta['selected'],
  'selected_rank':meta['rank'],
  'adapter_git_blob_sha256':{'stage26':hashlib.sha256((a.stage26_adapter/'adapter_model.safetensors').read_bytes()).hexdigest(),
    'stage31':hashlib.sha256((chosen/'adapter_model.safetensors').read_bytes()).hexdigest()},
  'counts':{'examples':128,'categories':8,'per_category':16,'models':3},
  'overall':overview,'by_category':category_scores,
  'within_source_stratified_bootstrap':uncertainties,'paired_ce':table,
  'time_s':time.perf_counter()-start,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30,
  'limitations':['128 new test identities never in Stage26 704 selected records but SAME underlying public Dolly source and model could have seen it in pretraining',
    'Reference-answer cross-entropy is not semantic correctness or human safety preference',
    'Stratified bootstrap conditional on fixed test selection and one pretrained model, does not capture training-seed variance',
    'No candidate chosen using this Stage34 test; winner frozen by Stage31 validation before evaluation']}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('STAGE34_UNSEEN_128_HOLDOUT_PASS',json.dumps(overview,indent=2),flush=True)
if __name__=='__main__':main()
