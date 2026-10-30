"""Real Qwen3-8B NF4 SFT from base with deterministic validated constraint-format rehearsal.

Equal 512 original-source IDs *2 epochs and identical model/optimizer parameters
across baseline mix0 and constrained 25%/50%. No official IFEval prompt is used
for training, nor any test response or generated teacher label.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,random,statistics,time
from pathlib import Path
from llm_opt_lab.training.stage26_real_sft import MODEL,collate,evaluate,load_split
RANK=16
SEEDS=(4011,4027)
MIXES=(0.,.25,.50)
EPOCHS=2
REPLAY_KINDS=('json_exact','xml_tag','begin_end')

def choose_replay_kind(index:int,epoch:int)->str:
 if not (0<=index<512 and epoch in (0,1)):raise ValueError("unregistered example index or epoch")
 return REPLAY_KINDS[(index+epoch)%len(REPLAY_KINDS)]

def encode(tokenizer,user,answer):
 prefix=tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=True,
  add_generation_prompt=True,enable_thinking=False)
 full=tokenizer.apply_chat_template([{'role':'user','content':user},{'role':'assistant','content':answer}],
  tokenize=True,add_generation_prompt=False,enable_thinking=False)
 if len(full)>384 or full[:len(prefix)]!=prefix or len(full)-len(prefix)<2:
  raise ValueError('format replay length/mask corrupt')
 return {'input_ids':full,'labels':[-100]*len(prefix)+full[len(prefix):],
   'answer_count':len(full)-len(prefix)}

def source_pairs(data:Path,tokenizer):
 manifest=json.loads((data/'stage40_manifest.json').read_text())
 payload=(data/'stage40_dolly_format_replay.jsonl').read_bytes()
 if hashlib.sha256(payload).hexdigest()!=manifest['dolly_format_replay_sha256']:
  raise ValueError('unapproved formatting source edit')
 originals=load_split(data/'stage26_train.jsonl',tokenizer)
 extra=[json.loads(s) for s in payload.decode().splitlines()]
 if len(extra)!=len(originals)!=512:raise ValueError('512 training items missing')
 if len(set(x['identity_sha256'] for x in extra))!=512:raise ValueError('source duplication')
 encoded=[]
 for base,record in zip(originals,extra):
  if base['identity_sha256']!=record['identity_sha256']:raise ValueError('source order mismatch')
  variants={}
  for kind,task in record['derived'].items():
   if kind not in REPLAY_KINDS:raise ValueError('unknown derived instruction')
   item=encode(tokenizer,task['user'],task['answer'])
   if len(item['input_ids'])!=task['total_tokens']:raise ValueError('derived token count drift')
   variants[kind]=item
  encoded.append({'identity_sha256':base['identity_sha256'],'raw':base,'variants':variants})
 return encoded,manifest

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--seed',type=int,choices=SEEDS,required=True)
 p.add_argument('--mix',type=float,choices=MIXES,required=True)
 p.add_argument('--data',type=Path,default=Path('data'))
 p.add_argument('--adapter',type=Path,required=True)
 p.add_argument('--out',type=Path,required=True)
 a=p.parse_args(argv)
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
 from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
 if not torch.cuda.is_available():raise RuntimeError('real CUDA device required')
 torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 dataset,manifest=source_pairs(a.data,tokenizer)
 val=load_split(a.data/'stage26_validation.jsonl',tokenizer)
 if len(val)!=96:raise ValueError('frozen Dolly val96 changed')
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,device_map={'':0},
  dtype=torch.bfloat16,quantization_config=q,attn_implementation='sdpa')
 base.config.use_cache=False
 base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
 model=get_peft_model(base,LoraConfig(r=RANK,lora_alpha=2*RANK,lora_dropout=0.,bias='none',
  target_modules=['q_proj','v_proj'],task_type=TaskType.CAUSAL_LM))
 optimizer=torch.optim.AdamW((par for par in model.parameters() if par.requires_grad),lr=1.5e-4)
 base_validation=evaluate(model,val,tokenizer.eos_token_id,limit=96)
 best=float('inf');best_epoch=None;history=[];times=[]
 seen_ids=[];torch.cuda.reset_peak_memory_stats()
 for epoch in range(EPOCHS):
  order=list(range(512));random.Random(a.seed+epoch*101).shuffle(order)
  fmt_indices=set(random.Random(a.seed+epoch*10007+4000).sample(range(512),int(round(512*a.mix))))
  used=collections.Counter();all_epoch_ids=set()
  for step in range(256):
   idxs=order[step*2:step*2+2]
   optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();step_loss=[]
   for index in idxs:
    entry=dataset[index]
    if entry['identity_sha256'] in all_epoch_ids:raise ValueError('duplicate within epoch')
    all_epoch_ids.add(entry['identity_sha256']);seen_ids.append(entry['identity_sha256'])
    training=entry['raw'];chosen='raw'
    if index in fmt_indices:
     kind=choose_replay_kind(index,epoch)
     training=entry['variants'].get(kind,entry['raw'])
     chosen=kind if kind in entry['variants'] else 'raw_overflow_fallback'
    used[chosen]+=1
    batch=collate([training],tokenizer.eos_token_id)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     loss=model(**batch,use_cache=False).loss
    if not torch.isfinite(loss):raise ValueError('nonfinite constrained SFT loss')
    (loss/2.).backward();step_loss.append(float(loss.detach()))
   torch.nn.utils.clip_grad_norm_((par for par in model.parameters() if par.requires_grad),1.)
   optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
   if step>=3:times.append(elapsed)
   if (step+1)%64==0:print('STAGE40_UPDATE','mix',a.mix,'seed',a.seed,'epoch',epoch+1,
      'step',step+1,'train_CE',round(statistics.mean(step_loss),4),flush=True)
  if len(all_epoch_ids)!=512:raise ValueError('not entire 512 real Dolly identities per epoch')
  validation=evaluate(model,val,tokenizer.eos_token_id,limit=96)
  record={'epoch':epoch+1,'unique_source_records':len(all_epoch_ids),
    'format_reference_demonstrations_used':sum(v for k,v in used.items() if k in REPLAY_KINDS),
    'format_distribution':dict(used),'validation_ce':validation['assistant_ce_nats'],
    'validation_ppl':validation['assistant_perplexity']}
  history.append(record)
  if record['validation_ce']<best:
   best=record['validation_ce'];best_epoch=epoch+1
   a.adapter.mkdir(parents=True,exist_ok=True)
   model.save_pretrained(a.adapter,safe_serialization=True)
  print('STAGE40_EPOCH_VALID',a.mix,a.seed,record,flush=True)
 if len(seen_ids)!=1024 or len(set(seen_ids))!=512:raise ValueError('two full source epochs not trained')
 metrics={'status':'measured','kind':'stage40_deterministic_dolly_constraint_rehearsal_qwen3_nf4',
  'config':{'mix':a.mix,'seed':a.seed,'rank':RANK,'training_epochs':EPOCHS,'optimizer_updates':512,
      'source_exposures':1024,'batch_accumulation':2,'optimizer':'AdamW','lr':.00015,'compute_precision':'BF16'},
  'format_source_sha256':manifest['dolly_format_replay_sha256'],
  'human_source_dataset_sha256':manifest['source_dolly_manifest_sha256'],
  'baseline_validation':base_validation,'per_epoch':history,
  'best_epoch_by_dolly_validation_only':best_epoch,'best_validation_ce':best,
  'speed':{'median_step_ms_excluding_first3_per_epoch':statistics.median(times)*1000,
    'samples_per_second_excluding_first3_per_epoch':2*len(times)/sum(times),
    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30},
  'adapter_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.adapter.glob('*') if x.is_file()},
  'caveats':['Derived format examples are algorithmically transformed from human-written Dolly responses, no human validation of new form',
    'All mixes maintain same original source identity exposures but assistant targets/contexts differ',
    'Single seed and public Dolly may overlap pretrained Qwen; final IFEval unseen source held out for independent evaluation']}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(metrics,indent=2)+'\n')
 print('STAGE40_REAL_TRAIN_COMPLETE',a.seed,a.mix,'DOLLY_VAL',round(best,5),'EPOCH',best_epoch,flush=True)
if __name__=='__main__':main()
