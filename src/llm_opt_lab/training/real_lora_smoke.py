"""Actual frozen Qwen2.5 causal LM LoRA adapter training on offline GPU."""
import argparse,json,hashlib,time,statistics,pathlib

def main():
 p=argparse.ArgumentParser();p.add_argument('--accum',type=int,default=1);p.add_argument('--steps',type=int,default=6);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--checkpoint-dir',type=pathlib.Path);p.add_argument('--checkpoint-every',type=int,default=2);a=p.parse_args()
 assert a.accum in (1,2,4) and a.steps>=3
 import torch
 from torch import nn
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.manual_seed(701);torch.cuda.manual_seed_all(701)
 repo='/home/you/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
 model=AutoModelForCausalLM.from_pretrained(repo,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
 tok=AutoTokenizer.from_pretrained(repo,local_files_only=True)
 for x in model.parameters():x.requires_grad_(False)
 class LoRALinear(nn.Module):
  def __init__(self,base,rank=8,alpha=16.):
   super().__init__(); self.base=base;self.scale=alpha/rank
   self.A=nn.Parameter(torch.randn(rank,base.in_features,device='cuda',dtype=torch.float32)*.01)
   self.B=nn.Parameter(torch.zeros(base.out_features,rank,device='cuda',dtype=torch.float32))
  def forward(self,x):
   z=self.base(x)
   adapter=(x.float()@self.A.T)@self.B.T
   return z+adapter.to(z.dtype)*self.scale
 n=0
 for layer in model.model.layers:
  for name in ('q_proj','v_proj'):
   setattr(layer.self_attn,name,LoRALinear(getattr(layer.self_attn,name)));n+=1
 trainable=[p for p in model.parameters() if p.requires_grad]
 opt=torch.optim.AdamW(trainable,lr=3e-4)
 prompts=['A practical purpose of gradient accumulation is to preserve an effective batch size.',
          'The RMSNorm kernel normalizes by the root of the mean squared activation.',
          'A schedulable LLM request should report queue waiting and token latency.',
          'Knowledge distillation can train smaller models using teacher outputs.']
 ids=[tok(t,return_tensors='pt',truncation=True,max_length=96)['input_ids'].to('cuda') for t in prompts]
 heldout=tok('In neural language models, attention can connect distant tokens.',return_tensors='pt')['input_ids'].to('cuda')
 def evaluate():
  model.eval()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   value=float(model(input_ids=heldout,labels=heldout,use_cache=False).loss)
  model.train()
  return value
 heldout_before=evaluate()
 torch.cuda.reset_peak_memory_stats();times=[];losses=[]
 model.train()
 from llm_opt_lab.training.async_checkpoint import AsyncCheckpointWriter
 writer=AsyncCheckpointWriter(a.checkpoint_dir) if a.checkpoint_dir else None
 saved_steps=[]
 for step in range(a.steps):
  opt.zero_grad(set_to_none=True);torch.cuda.synchronize();t0=time.perf_counter()
  for micro in range(a.accum):
   x=ids[(step*a.accum+micro)%len(ids)]
   with torch.autocast('cuda',dtype=torch.bfloat16):
    out=model(input_ids=x,labels=x,use_cache=False)
    loss=out.loss/a.accum
   loss.backward()
  opt.step();torch.cuda.synchronize()
  if writer and (step+1)%a.checkpoint_every==0:
   # Bound the pipeline and surface errors rather than dropping checkpoint.
   writer.flush()
   writer.submit(step+1,{n:p for n,p in model.named_parameters() if p.requires_grad})
   saved_steps.append(step+1)
  if step>=2:
   times.append(time.perf_counter()-t0)
   losses.append(float(loss.detach())*a.accum)
 if writer:writer.close()
 heldout_after=evaluate()
 from safetensors.torch import save_file
 adapter_path=a.out.with_suffix('.safetensors')
 weights={n:x.detach().cpu().contiguous() for n,x in model.named_parameters() if x.requires_grad}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 save_file(weights,str(adapter_path),metadata={'format':'pt','base_model':'Qwen2.5-1.5B-Instruct','rank':'8','seed':'701'})
 digest=hashlib.sha256(adapter_path.read_bytes()).hexdigest()
 result={'status':'measured' ,'model':'Qwen2.5-1.5B-Instruct FP16-weight source loaded BF16','kind':'real_frozen_transformer_lora_smoke','modules_wrapped':n,'trainable_parameters':sum(x.numel() for x in trainable),'effective_sequences':a.accum,'steps_measured':len(times),'median_step_ms':statistics.median(times)*1000,'steps_per_sec':len(times)/sum(times),'peak_allocated_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'last_train_loss':losses[-1],'heldout_loss_before':heldout_before,'heldout_loss_after':heldout_after,'adapter_file':adapter_path.name,'adapter_sha256':digest,'async_checkpoint_steps':saved_steps, 'caveats':['Tiny demonstration text; no downstream quality proof','LoRA-only trainable low-rank matrices, frozen base','Not Qwen3-8B QLoRA; not representative of production 8B post-training']}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
