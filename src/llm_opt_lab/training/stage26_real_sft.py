"""Real Dolly 15k Qwen3-8B NF4 SFT: assistant-only CE, frozen splits, validation/test."""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics,time
from pathlib import Path

MODEL='/home/you/models/Qwen3-8B'

def load_split(path,tokenizer,max_len=384):
 rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
 out=[]
 for r in rows:
  user=r['instruction']+('\n\nReference context:\n'+r['context'] if r['context'] else '')
  prompts=[{'role':'user','content':user}]
  prefix=tokenizer.apply_chat_template(prompts,tokenize=True,add_generation_prompt=True,enable_thinking=False)
  full=tokenizer.apply_chat_template(prompts+[{'role':'assistant','content':r['response']}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
  if full[:len(prefix)]!=prefix or len(full)>max_len or len(full)-len(prefix)<2:
   raise ValueError('data split/chat masking is invalid')
  labels=[-100]*len(prefix)+full[len(prefix):]
  out.append({'input_ids':full,'labels':labels,'answer_count':len(full)-len(prefix),
     'identity_sha256':r['identity_sha256'],'category':r['category'], 'user':user,'reference':r['response']})
 return out

def collate(items,pad_id):
 import torch
 n=max(len(x['input_ids']) for x in items)
 ids=torch.tensor([x['input_ids']+[pad_id]*(n-len(x['input_ids'])) for x in items],dtype=torch.long,device='cuda')
 labels=torch.tensor([x['labels']+[-100]*(n-len(x['labels'])) for x in items],dtype=torch.long,device='cuda')
 attention=torch.tensor([[1]*len(x['input_ids'])+[0]*(n-len(x['input_ids'])) for x in items],dtype=torch.long,device='cuda')
 if not torch.all((labels!=-100).sum(dim=1)>0):raise ValueError('empty assistant target')
 return dict(input_ids=ids,attention_mask=attention,labels=labels)

def evaluate(model,records,pad_id,limit=None):
 import torch
 model.eval();neglog=0.;target_tokens=0;losses=[]
 subset=records[:limit] if limit else records
 with torch.no_grad():
  for row in subset:
   b=collate([row],pad_id)
   result=model(**b,use_cache=False)
   l=float(result.loss.detach())
   if not math.isfinite(l):raise ValueError('nonfinite validation loss')
   tokens=row['answer_count'];neglog+=l*tokens;target_tokens+=tokens;losses.append(l)
 model.train()
 mean=neglog/target_tokens
 return {'count':len(subset),'target_tokens':target_tokens,'assistant_ce_nats':mean,
         'assistant_perplexity':math.exp(min(mean,30)),'mean_record_loss':statistics.mean(losses)}

def sample_generate(model,tokenizer,records,n=12):
 import torch
 model.eval();samples=[]
 with torch.inference_mode():
  for r in records[:n]:
   ids=tokenizer.apply_chat_template([{'role':'user','content':r['user']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
   x=torch.tensor([ids],device='cuda')
   generated=model.generate(input_ids=x,max_new_tokens=80,do_sample=False,pad_token_id=tokenizer.eos_token_id)
   text=tokenizer.decode(generated[0,len(ids):],skip_special_tokens=True)
   reference=r['reference']
   def tokens(t):return set(t.lower().split())
   ref=tokens(reference);pred=tokens(text)
   precision=len(ref&pred)/max(1,len(pred));recall=len(ref&pred)/max(1,len(ref))
   f1=2*precision*recall/(precision+recall) if precision+recall>0 else 0.
   samples.append({'id_sha256':r['identity_sha256'],'category':r['category'],
          'reference_length':len(reference),'generated_length':len(text),'token_overlap_f1':f1,
          'generation_sha256':hashlib.sha256(text.encode()).hexdigest()})
 model.train()
 return {'count':len(samples),'mean_reference_token_overlap_f1':statistics.mean(x['token_overlap_f1'] for x in samples),'per_case':samples,
   'note':'lexical overlap is a weak reference metric, not semantic quality or alignment; no reference texts saved in result'}

def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--action',choices=('train','test'),default='train')
 ap.add_argument('--rank',type=int,choices=(8,16),default=8)
 ap.add_argument('--micro-batch',type=int,choices=(1,2),default=1)
 ap.add_argument('--accum',type=int,choices=(1,2),default=2)
 ap.add_argument('--precision',choices=('bf16','fp16'),default='bf16')
 ap.add_argument('--grad-checkpoint',action='store_true')
 ap.add_argument('--steps',type=int,default=24)
 ap.add_argument('--seed',type=int,default=2610)
 ap.add_argument('--input',type=Path,default=Path('data'))
 ap.add_argument('--adapter',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 a=ap.parse_args(argv)
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training,PeftModel
 if not torch.cuda.is_available():raise RuntimeError('CUDA only; do not fabricate GPU metrics')
 if a.steps<4:raise ValueError('not enough train steps')
 manifest=json.loads((a.input/'stage26_manifest.json').read_text())
 for split in ('train','validation','test'):
  body=(a.input/f'stage26_{split}.jsonl').read_bytes()
  if hashlib.sha256(body).hexdigest()!=manifest['splits'][split]['sha256']:
   raise ValueError('frozen data split hash drift '+split)
 torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 dtype=torch.bfloat16 if a.precision=='bf16' else torch.float16
 quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 val=load_split(a.input/'stage26_validation.jsonl',tokenizer)
 test=load_split(a.input/'stage26_test.jsonl',tokenizer) if a.action=='test' else []
 train=load_split(a.input/'stage26_train.jsonl',tokenizer) if a.action=='train' else []
 base=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,device_map={'':0},dtype=dtype,
      quantization_config=quant,attn_implementation='sdpa')
 base.config.use_cache=False
 if a.action=='train':
  base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=a.grad_checkpoint,
      gradient_checkpointing_kwargs={'use_reentrant':False} if a.grad_checkpoint else None)
  model=get_peft_model(base,LoraConfig(r=a.rank,lora_alpha=2*a.rank,lora_dropout=0.,bias='none',
                       target_modules=['q_proj','v_proj'],task_type=TaskType.CAUSAL_LM))
  if a.grad_checkpoint: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 else:
  if not a.adapter.exists():raise FileNotFoundError('no frozen selected adapter')
  model=PeftModel.from_pretrained(base,a.adapter,is_trainable=False)
 trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
 model.train() if a.action=='train' else model.eval()
 torch.cuda.reset_peak_memory_stats()
 before=evaluate(model,val,tokenizer.eos_token_id,limit=48)
 result={'status':'measured','kind':'stage26_qwen3_8b_nf4_real_dolly_sft',
    'action':a.action,'dataset_sha256':manifest['source_sha256'],
    'splits':{k:v['sha256'] for k,v in manifest['splits'].items()},
    'config':{'rank':a.rank,'micro_batch':a.micro_batch,'accum':a.accum,
              'effective_batch':a.micro_batch*a.accum,'precision':a.precision,
              'grad_checkpoint':a.grad_checkpoint,'steps':a.steps,'seed':a.seed},
    'trainable_parameters':trainable,'validation_before':before,
    'notes':['Human-written Databricks Dolly records, answer-only masked CE','Small training budget; no full alignment benchmark',
             'Different-rank experiments have different optimizer capacity; compare with same data/steps']}
 if a.action=='train':
  opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1.5e-4)
  scaler=torch.amp.GradScaler('cuda',enabled=(a.precision=='fp16'))
  rng=random.Random(a.seed)
  order=list(range(len(train)));rng.shuffle(order);pointer=0;times=[];loss_history=[]
  for step in range(a.steps):
   opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
   mb_losses=[]
   for micro in range(a.accum):
    if pointer+a.micro_batch>len(order):rng.shuffle(order);pointer=0
    batch=[train[order[j]] for j in range(pointer,pointer+a.micro_batch)]
    pointer+=a.micro_batch
    data=collate(batch,tokenizer.eos_token_id)
    with torch.autocast('cuda',dtype=dtype):
     loss=model(**data,use_cache=False).loss
    if not torch.isfinite(loss):raise ValueError('invalid training loss')
    mb_losses.append(float(loss.detach()))
    scaler.scale(loss/a.accum).backward()
   scaler.unscale_(opt)
   torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
   scaler.step(opt);scaler.update();torch.cuda.synchronize()
   dt=time.perf_counter()-start
   if step>=3:
    times.append(dt);loss_history.append(statistics.mean(mb_losses))
   if step%8==0:print('SFT_STEP',step+1,'loss',round(statistics.mean(mb_losses),4),'time_s',round(dt,4),flush=True)
  after=evaluate(model,val,tokenizer.eos_token_id,limit=48)
  model.save_pretrained(a.adapter,safe_serialization=True)
  result.update(validation_after=after,training={'warmup_steps':3,'measured_steps':len(times),
      'mean_train_loss':statistics.mean(loss_history),'final_train_loss':loss_history[-1],
      'median_step_ms':statistics.median(times)*1000,
      'sample_throughput_per_second':len(times)*a.micro_batch*a.accum/sum(times),
      'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30},
      adapter_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.adapter.iterdir() if p.is_file()})
 else:
  final=evaluate(model,test,tokenizer.eos_token_id,limit=96)
  result.update(test_metrics=final,validation_after=before,
                generated_samples=sample_generate(model,tokenizer,test,n=12),
                final_report_note='Frozen test evaluated only after selecting adapter from validation.')
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(result,indent=2)+'\n')
 print('STAGE26_RESULT',a.action,result.get('validation_after'),result.get('test_metrics'),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
