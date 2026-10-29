"""Real Qwen3-8B NF4 QLoRA: 2 FULL shuffled epochs of all 512 Dolly train IDs.

Compare two ranks/two independent init+shuffle seeds; save checkpoints by
validation only. Test is forbidden here and run once after validation selection.
"""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics,time
from pathlib import Path
from llm_opt_lab.training.stage26_real_sft import MODEL,load_split,collate,evaluate

ALLOWED_SEEDS=(3111,3127)
ALLOWED_RANKS=(8,16)
STEPS_PER_EPOCH=256
EPOCHS=2

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--rank',type=int,choices=ALLOWED_RANKS,required=True)
 p.add_argument('--seed',type=int,choices=ALLOWED_SEEDS,required=True)
 p.add_argument('--input',type=Path,default=Path('data'))
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--adapter',type=Path,required=True)
 a=p.parse_args(argv)
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
 if not torch.cuda.is_available():raise RuntimeError('GPU-only real training')
 manifest=json.loads((a.input/'stage26_manifest.json').read_text())
 for split in ('train','validation','test'):
  h=hashlib.sha256((a.input/f'stage26_{split}.jsonl').read_bytes()).hexdigest()
  if h!=manifest['splits'][split]['sha256']:raise ValueError('frozen split integrity failed: '+split)
 torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 train=load_split(a.input/'stage26_train.jsonl',t)
 val=load_split(a.input/'stage26_validation.jsonl',t)
 if len(train)!=512 or len(val)!=96 or len({x['identity_sha256'] for x in train})!=512:raise ValueError('untrusted split')
 qconfig=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
  bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
 base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,device_map={'':0},
   dtype=torch.bfloat16,quantization_config=qconfig,attn_implementation='sdpa')
 base.config.use_cache=False
 base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
 model=get_peft_model(base,LoraConfig(r=a.rank,lora_alpha=2*a.rank,lora_dropout=0.,bias='none',
  target_modules=['q_proj','v_proj'],task_type=TaskType.CAUSAL_LM))
 nparams=sum(x.numel() for x in model.parameters() if x.requires_grad)
 optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=1.5e-4)
 baseline=evaluate(model,val,t.eos_token_id,limit=96)
 best_score=float('inf');best_epoch=0;per_epoch=[]
 all_times=[];all_losses=[];observed_ids=[]
 torch.cuda.reset_peak_memory_stats()
 rng=random.Random(a.seed)
 for epoch in range(EPOCHS):
  order=list(range(len(train)));rng.shuffle(order)
  seen=set()
  timings=[];training_ce=[]
  for step in range(STEPS_PER_EPOCH):
   samples=[train[order[step*2]],train[order[step*2+1]]]
   for s in samples:
    if s['identity_sha256'] in seen:raise ValueError('duplicate record within shuffled epoch')
    seen.add(s['identity_sha256']);observed_ids.append(s['identity_sha256'])
   optimizer.zero_grad(set_to_none=True)
   torch.cuda.synchronize();start=time.perf_counter()
   losses=[]
   for row in samples:
    batch=collate([row],t.eos_token_id)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     loss=model(**batch,use_cache=False).loss
    if not torch.isfinite(loss):raise ValueError('nonfinite train loss')
    (loss/2.).backward()
    losses.append(float(loss.detach()))
   torch.nn.utils.clip_grad_norm_((x for x in model.parameters() if x.requires_grad),1.)
   optimizer.step()
   torch.cuda.synchronize()
   duration=time.perf_counter()-start
   all_times.append(duration);all_losses.append(statistics.mean(losses))
   timings.append(duration);training_ce.append(statistics.mean(losses))
   if (step+1)%64==0:
    print('STAGE31_STEP',a.rank,a.seed,'epoch',epoch+1,'step',step+1,
      'loss',round(statistics.mean(losses),4),'gpu_peak_gib',round(torch.cuda.max_memory_allocated()/2**30,3),flush=True)
  if len(seen)!=512:raise ValueError('full data coverage incomplete')
  validation=evaluate(model,val,t.eos_token_id,limit=96)
  entry={'epoch':epoch+1,'updates':(epoch+1)*STEPS_PER_EPOCH,'train_records_seen_unique':len(seen),
    'reference_train_loss_mean':statistics.mean(training_ce),
    'validation':validation,'measured_median_step_ms':statistics.median(timings)*1000,
    'measured_samples_per_second':len(timings)*2/sum(timings)}
  per_epoch.append(entry)
  if validation['assistant_ce_nats']<best_score:
   best_score=validation['assistant_ce_nats'];best_epoch=epoch+1
   a.adapter.mkdir(parents=True,exist_ok=True)
   model.save_pretrained(a.adapter,safe_serialization=True)
  print('STAGE31_EPOCH_VALID',a.rank,a.seed,'epoch',epoch+1,
    'val_ce',round(validation['assistant_ce_nats'],5),'best_epoch',best_epoch,flush=True)
 if len(observed_ids)!=1024 or len(set(observed_ids))!=512:raise ValueError('not exactly two full 512-ID epochs')
 out={'status':'measured','kind':'stage31_real_dolly512_two_epochs_qwen3_8b_qlora',
  'config':{'rank':a.rank,'seed':a.seed,'epochs':EPOCHS,'steps_per_epoch':256,'total_updates':512,
     'effective_batch':2,'micro_batch':1,'accumulation':2,'precision':'bf16','gradient_checkpoint':False,
     'optimizer':'AdamW','lr':.00015},
  'dataset_sha256':manifest['source_sha256'],
  'source_splits':{k:v['sha256'] for k,v in manifest['splits'].items()},
  'trainable_parameters':nparams,'train_unique_records':512,
  'train_exposures':len(observed_ids),'train_epochs':per_epoch,'baseline_validation':baseline,
  'selected_by_validation_epoch':best_epoch,'best_validation_ce':best_score,
  'training_efficiency':{'median_update_ms_excluding_first5':statistics.median(all_times[5:])*1000,
    'samples_per_second_excluding_first5':(len(all_times)-5)*2/sum(all_times[5:]),
    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30},
  'adapter_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.adapter.glob('*')) if f.is_file()},
  'limitations':['512 distinct training records from licensed human-authored Dolly but potentially in pretrained corpus',
     'Only two epochs and two seeds; do not assume repeatability beyond measured configurations',
     'No frozen test is evaluated by this training script; select candidate by validation only']}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(out,indent=2)+'\n')
 print('STAGE31_SUCCESS',a.rank,a.seed,'best',best_score,'epoch',best_epoch,'seen',len(observed_ids),flush=True)
if __name__=='__main__':main()
