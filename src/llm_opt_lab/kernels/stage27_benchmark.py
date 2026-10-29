"""Qwen3-8B operator-shape benchmark vs eager and torch.compile + torch profiler."""
from __future__ import annotations
import argparse,json,math,statistics,time
from pathlib import Path

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--warmup-ms',type=int,default=100)
 p.add_argument('--rep-ms',type=int,default=240)
 p.add_argument('--profile',action='store_true')
 a=p.parse_args(argv)
 import torch,triton,triton.testing
 from torch.profiler import ProfilerActivity,profile
 from llm_opt_lab.kernels.stage27_transformer_ops import fused_residual_weighted_rmsnorm,causal_materialized_softmax
 if not torch.cuda.is_available():raise RuntimeError('real NVIDIA GPU required')
 import hashlib
 config=json.load(open('/home/you/models/Qwen3-8B/config.json'))
 assert config['hidden_size']==4096 and config['num_attention_heads']==32 and config['rms_norm_eps']==1e-6
 torch.manual_seed(27021);torch.cuda.manual_seed_all(27021)
 results=[];profiles={}
 def benchmark(label,description,fn_eager,fn_compiled,fn_triton,atol,rtol):
  expected=fn_eager();actual=fn_triton();compiled=fn_compiled()
  if not isinstance(expected,tuple):expected=(expected,);actual=(actual,);compiled=(compiled,)
  assert len(expected)==len(actual)==len(compiled)
  error=0.
  for e,c,t in zip(expected,compiled,actual):
   torch.testing.assert_close(t,e,atol=atol,rtol=rtol)
   torch.testing.assert_close(c,e,atol=atol,rtol=rtol)
   error=max(error,(t.float()-e.float()).abs().max().item())
  timings={}
  for key,fn in [('pytorch_eager',fn_eager),('torch_compile',fn_compiled),('custom_triton',fn_triton)]:
   samples=[]
   for rep in range(3):
    median=triton.testing.do_bench(fn,warmup=a.warmup_ms,rep=a.rep_ms,quantiles=[.5])[0]
    samples.append(float(median))
   timings[key]={'median_ms':statistics.median(samples),'repeats_ms':samples}
  tri=timings['custom_triton']['median_ms'];compiletime=timings['torch_compile']['median_ms'];eager=timings['pytorch_eager']['median_ms']
  entry={'name':label,'workload':description,'numerical_max_abs_error':error,'atol':atol,'rtol':rtol,
         'timings':timings,'speedup_vs_compile':compiletime/tri,'speedup_vs_eager':eager/tri}
  if a.profile and label in ('fused_rms_rows256_bfloat16','masked_softmax_seq512'):
   operator_profiles={}
   for key,fn in [('torch_compile',fn_compiled),('custom_triton',fn_triton)]:
    for _ in range(3):fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:
     for _ in range(12):fn()
     torch.cuda.synchronize()
    averages=[]
    for item in prof.key_averages():
     gpu=float(getattr(item,'self_device_time_total',getattr(item,'self_cuda_time_total',0)))
     if gpu>0:averages.append({'name':str(item.key),'calls':int(item.count),'gpu_self_us':gpu})
    operator_profiles[key]=sorted(averages,key=lambda x:x['gpu_self_us'],reverse=True)[:12]
   profiles[label]=operator_profiles
  print('STAGE27_CASE',label,'compile_ms',round(compiletime,4),'triton_ms',round(tri,4),'ratio',round(compiletime/tri,3),flush=True)
  results.append(entry)
 def rms_reference(x,r,g):
  z=(x.float()+r.float()).to(x.dtype)
  norm=(z.float()*torch.rsqrt(z.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
  return norm*g,z
 for dtype in (torch.float16,torch.bfloat16):
  for rows in (64,256,1024):
   cols=4096
   x=torch.randn((rows,cols),device='cuda',dtype=dtype)
   r=torch.randn_like(x)*.1
   gamma=torch.ones((cols,),device='cuda',dtype=dtype)+torch.randn((cols,),device='cuda',dtype=dtype)*.01
   def eager():return rms_reference(x,r,gamma)
   compiled=torch.compile(lambda x,r,g:rms_reference(x,r,g),backend='inductor',fullgraph=True)
   def comp():return compiled(x,r,gamma)
   def trit():return fused_residual_weighted_rmsnorm(x,r,gamma)
   benchmark(f'fused_rms_rows{rows}_{str(dtype).split(".")[-1]}',f'Qwen3 hidden={cols} rows={rows} dtype={dtype}',
             eager,comp,trit,.016 if dtype==torch.bfloat16 else .004,.016 if dtype==torch.bfloat16 else .004)
 for seq in (256,512):
  heads=32;dtype=torch.bfloat16
  scores=torch.randn((heads*seq,seq),device='cuda',dtype=dtype)
  mask=torch.arange(seq,device='cuda').view(1,seq)>torch.arange(seq,device='cuda').view(seq,1)
  mask=mask.repeat(heads,1)
  def ref(t):return torch.softmax(t.masked_fill(mask,-float('inf')).float(),dim=-1).to(t.dtype)
  compiled=torch.compile(ref,backend='inductor',fullgraph=True)
  benchmark(f'masked_softmax_seq{seq}',f'Qwen3 heads=32 causal attention score MATERIALIZED shape={scores.shape}; NOT FlashAttention',
       lambda:ref(scores),lambda:compiled(scores),lambda:causal_materialized_softmax(scores,seq),.016,.016)
 outcome={'status':'measured','kind':'stage27_qwen3_transformer_operator_optimization',
  'reference_model_config':{'hidden_size':4096,'attention_heads':32,'layers':36,'rms_norm_eps':1e-6},
  'timing':{'warmup_ms':a.warmup_ms,'rep_ms':a.rep_ms,'repeats':3},
  'cases':results,'profiler':profiles,
  'limitations':['Qwen3 shapes and semantics, no end-to-end transformer swap','Materialized softmax is not FlashAttention','No Nsight Compute DRAM/occupancy hardware counters','torch.compile excludes compilation cost; repeat timing noise possible']}
 a.out.parent.mkdir(parents=True,exist_ok=True)
 a.out.write_text(json.dumps(outcome,indent=2)+'\n')
 print('STAGE27_DONE',len(results),'cases',flush=True)
if __name__=='__main__':raise SystemExit(main())
