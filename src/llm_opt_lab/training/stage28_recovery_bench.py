"""Actual frozen Transformer LoRA exact resume from async optimizer+RNG checkpoints."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,time
from pathlib import Path
from llm_opt_lab.training.stage28_checkpoint import AtomicTrainingCheckpoint,verify_load

MODEL='/home/you/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
TEXTS=['A language model stores attention keys and values in a KV cache.',
 'Training low-rank adapters can reduce the number of trainable parameters.',
 'An asynchronous checkpoint cannot be reported complete before its manifest is written.',
 'Automatic mixed precision trades numerical range against computational throughput.',
 'Serving a transformer efficiently requires separating prefill and decode costs.',
 'A confidence interval should account for independent workloads and stochastic variation.']

def init_model(seed):
 import torch
 from torch import nn
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
                                           attn_implementation='sdpa').cuda()
 for p in model.parameters():p.requires_grad_(False)
 class LowRank(nn.Module):
  def __init__(self,base):
   super().__init__();self.base=base;self.scale=2.
   self.A=nn.Parameter(torch.randn((8,base.in_features),device='cuda',dtype=torch.float32)*.01)
   self.B=nn.Parameter(torch.zeros((base.out_features,8),device='cuda',dtype=torch.float32))
  def forward(self,x):return self.base(x)+(((x.float()@self.A.T)@self.B.T)*self.scale).to(x.dtype)
 for layer in model.model.layers:
  layer.self_attn.q_proj=LowRank(layer.self_attn.q_proj)
  layer.self_attn.v_proj=LowRank(layer.self_attn.v_proj)
 params=[p for p in model.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(params,lr=2e-4)
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 data=[tok(x,return_tensors='pt')['input_ids'].cuda() for x in TEXTS]
 generator=torch.Generator(device='cpu').manual_seed(seed+9001)
 return model,optimizer,data,generator

def state_at_step(step,model,opt,gen):
 import torch
 return {'step':step,'trainable':{name:p for name,p in model.named_parameters() if p.requires_grad},
         'optimizer':opt.state_dict(),'data_rng':gen.get_state(),
         'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}

def restore(checkpoint,model,opt,gen):
 import torch
 state=verify_load(checkpoint)
 params={name:p for name,p in model.named_parameters() if p.requires_grad}
 if set(params)!=set(state['trainable']):raise ValueError('mismatched adapter names')
 with torch.no_grad():
  for name,p in params.items():p.copy_(state['trainable'][name].to('cuda'))
 opt.load_state_dict(state['optimizer'])
 gen.set_state(state['data_rng'])
 torch.set_rng_state(state['torch_rng'])
 torch.cuda.set_rng_state_all(state['cuda_rng'])
 return state['step']

def train_run(label,root,save_mode='none',split=False,seed=28008):
 import torch
 model,opt,data,gen=init_model(seed)
 import torch
 torch.cuda.reset_peak_memory_stats()
 writer=AtomicTrainingCheckpoint(root/label,async_write=save_mode=='async') if save_mode!='none' else None
 start_step=0;losses=[];step_ms=[];snap_ms=[]
 for step in range(12):
  if split and step==6:
   if writer:writer.close()
   del model,opt
   torch.cuda.empty_cache()
   model,opt,data,gen=init_model(seed)
   start_step=restore(root/label/'step_000006.pt',model,opt,gen)
   assert start_step==6
   writer=None
  idx=int(torch.randint(0,len(data),(1,),generator=gen).item())
  ids=data[idx]
  torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   loss=model(input_ids=ids,labels=ids,use_cache=False).loss
  if not torch.isfinite(loss):raise ValueError('invalid training CE')
  loss.backward();opt.step();torch.cuda.synchronize()
  if writer and (step+1)%3==0:
   writer.flush()
   copy_time=writer.submit(step+1,state_at_step(step+1,model,opt,gen))
   snap_ms.append(copy_time)
  torch.cuda.synchronize()
  elapsed=(time.perf_counter()-start)*1000
  if step>=2:step_ms.append(elapsed)
  losses.append(float(loss.detach()))
 if writer:writer.close()
 state=state_at_step(12,model,opt,gen)
 # Canonical adapter hash and optimizer tensors for deterministic recovery audit.
 dig=hashlib.sha256()
 for name,t in sorted(state['trainable'].items()):dig.update(name.encode()+t.detach().cpu().numpy().tobytes())
 adapter_hash=dig.hexdigest()
 optim=opt.state_dict();h=hashlib.sha256()
 for k,item in sorted(optim['state'].items()):
  h.update(str(k).encode())
  for key,val in sorted(item.items()):
   h.update(key.encode())
   if isinstance(val,torch.Tensor):h.update(val.detach().cpu().contiguous().numpy().tobytes())
   else:h.update(str(val).encode())
 return {'label':label,'steps':12,'loss_final':losses[-1],'adapter_hash':adapter_hash,
  'optimizer_hash':h.hexdigest(),'median_step_ms':statistics.median(step_ms),
  'avg_step_ms':statistics.mean(step_ms),'checkpoint_synchronous_copy_ms':snap_ms,
  'training_samples_per_second':1000/statistics.mean(step_ms),
  'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
  'restarted_from_step':start_step}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 import torch
 if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
 torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True)
 root=a.out.parent/'stage28_checkpoints';root.mkdir(parents=True,exist_ok=False)
 variants=[train_run('continuous',root),train_run('sync_checkpoints',root,'sync'),
           train_run('async_checkpoints',root,'async'),train_run('recovered_async',root,'async',split=True)]
 base=variants[0]
 for x in variants[1:]:
  if x['adapter_hash']!=base['adapter_hash'] or x['optimizer_hash']!=base['optimizer_hash']:
   raise ValueError('not exactly reproducible after checkpoint: '+x['label'])
  if x['loss_final']!=base['loss_final']:raise ValueError('final loss drift')
 # Verify all expected checkpoints and SHA, including restart point.
 evidence=[]
 for sub in ('sync_checkpoints','async_checkpoints','recovered_async'):
  folder=root/sub
  for p in sorted(folder.glob('step_*.pt')):
   state=verify_load(p)
   evidence.append({'case':sub,'step':state['step'],'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 out={'status':'pass','kind':'real_transformer_full_optimizer_rng_async_restart_v1',
      'model':'Qwen2.5-1.5B-Instruct BF16, LoRA Q/V rank8','exact_final_adapter_and_optimizer':True,
      'runs':variants,'checkpoints_verified':evidence,
      'limitations':['Small real Transformer adapter workload, not distributed or Qwen3-8B QLoRA','CPU snapshot is synchronous and included in timing; writer is asynchronous only for disk serialization',
                     'Single hardware / no repeated independent sessions']}
 a.out.write_text(json.dumps(out,indent=2)+'\n')
 print('STAGE28_EXACT_RESTART_PASS',len(variants),'runs',len(evidence),'checkpoint_files',flush=True)
 for v in variants:print(v['label'],round(v['median_step_ms'],2),v['adapter_hash'][:12],flush=True)
if __name__=='__main__':main()
