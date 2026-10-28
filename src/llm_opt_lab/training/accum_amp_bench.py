"""Measured single-GPU mixed precision/gradient accumulation training comparison."""
import argparse,json,time,statistics,pathlib

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fp32','bf16','fp16'],required=True);p.add_argument('--accum',type=int,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 import torch
 from torch import nn
 assert torch.cuda.is_available() and a.accum>0
 torch.manual_seed(17017);torch.cuda.manual_seed_all(17017)
 device='cuda';m=nn.Sequential(nn.Linear(1024,2048),nn.GELU(),nn.Linear(2048,1024)).to(device)
 opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
 scaler=torch.amp.GradScaler('cuda',enabled=a.mode=='fp16')
 dtype={'fp32':torch.float32,'bf16':torch.bfloat16,'fp16':torch.float16}[a.mode]
 gen=torch.Generator(device=device).manual_seed(17017)
 samples=0;times=[];losses=[];torch.cuda.reset_peak_memory_stats()
 for step in range(15):
  opt.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
  for micro in range(a.accum):
   x=torch.randn((16,1024),device=device,generator=gen)
   with torch.autocast('cuda',dtype=dtype,enabled=a.mode!='fp32'):
    y=m(x);loss=(y-0.4*x).square().mean()/a.accum
   scaler.scale(loss).backward()
  scaler.step(opt);scaler.update();torch.cuda.synchronize()
  elapsed=time.perf_counter()-start
  if step>=3:times.append(elapsed);losses.append(float(loss.detach())*a.accum);samples+=16*a.accum
 result={'status':'measured','mode':a.mode,'accum':a.accum,'effective_batch':16*a.accum,'measured_steps':12,'median_step_ms':statistics.median(times)*1000,'throughput_samples_per_s':samples/sum(times),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'final_loss':losses[-1], 'notes':'synthetic small MLP; not an LLM training or distributed result'}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
