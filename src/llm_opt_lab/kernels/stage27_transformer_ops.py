"""Real Qwen3-shaped fused residual+weighted RMSNorm and materialized causal softmax.

The second operator is a diagnostic only: production Qwen3 uses FlashAttention,
which avoids materializing the quadratic attention score tensor. Do not claim
this standalone softmax accelerates FlashAttention or end-to-end Qwen3 inference.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_residual_rms_kernel(X,R,G,Y,Z,N:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0)
 cols=tl.arange(0,BLOCK)
 m=cols<N
 x=tl.load(X+row*N+cols,mask=m,other=0).to(tl.float32)
 r=tl.load(R+row*N+cols,mask=m,other=0).to(tl.float32)
 # Preserve the torch reference's intermediate activation dtype rounding.
 z=(x+r).to(Y.dtype.element_ty)
 zn=z.to(tl.float32)
 std=tl.rsqrt(tl.sum(tl.where(m,zn*zn,0.),axis=0)/N+EPS)
 weight=tl.load(G+cols,mask=m,other=1).to(Y.dtype.element_ty)
 normalized=(zn*std).to(Y.dtype.element_ty)
 out=(normalized.to(tl.float32)*weight.to(tl.float32)).to(Y.dtype.element_ty)
 tl.store(Y+row*N+cols,out,mask=m)
 tl.store(Z+row*N+cols,z,mask=m)

@triton.jit
def _causal_masked_softmax_kernel(X,Y,N:tl.constexpr,BLOCK:tl.constexpr):
 row=tl.program_id(0)
 cols=tl.arange(0,BLOCK)
 query=row%N
 x=tl.load(X+row*N+cols,mask=cols<N,other=-float('inf')).to(tl.float32)
 x=tl.where(cols<=query,x,-float('inf'))
 z=tl.exp(x-tl.max(x,axis=0))
 y=z/tl.sum(z,axis=0)
 tl.store(Y+row*N+cols,y,mask=cols<N)

def fused_residual_weighted_rmsnorm(x,residual,gamma,eps=1e-6):
 if x.device.type!='cuda' or x.ndim!=2 or not x.is_contiguous():raise ValueError('CUDA row-major matrix required')
 if not residual.is_contiguous() or residual.shape!=x.shape or residual.dtype!=x.dtype:raise ValueError('residual mismatch')
 if gamma.shape!=(x.shape[1],) or gamma.device!=x.device or gamma.dtype!=x.dtype:raise ValueError('gamma mismatch')
 if x.dtype not in (torch.float16,torch.bfloat16) or eps<=0 or x.shape[1]>8192:raise ValueError('invalid dtype/dimension/eps')
 y=torch.empty_like(x);z=torch.empty_like(x)
 block=triton.next_power_of_2(x.shape[1]);warps=4 if block<=4096 else 8
 _fused_residual_rms_kernel[(x.shape[0],)](x,residual,gamma,y,z,x.shape[1],eps,block,num_warps=warps)
 return y,z

def causal_materialized_softmax(x,seq_len):
 if x.device.type!='cuda' or x.ndim!=2 or not x.is_contiguous():raise ValueError('CUDA row-major matrix required')
 if x.shape[1]!=seq_len or x.shape[0]%seq_len or seq_len<2 or seq_len>2048:raise ValueError('non-square head/query sequence')
 if x.dtype not in (torch.float16,torch.bfloat16):raise ValueError('unsupported dtype')
 out=torch.empty_like(x)
 _causal_masked_softmax_kernel[(x.shape[0],)](x,out,seq_len,triton.next_power_of_2(seq_len),num_warps=4)
 return out
