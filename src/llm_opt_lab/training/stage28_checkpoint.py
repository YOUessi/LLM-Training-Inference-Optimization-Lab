"""Bounded atomic async training-state snapshots, including optimizer & RNG.

Only trusted locally generated CPU tensor containers are serialized. Loading
requires torch.load(...,weights_only=True) plus manifest/SHA verification.
"""
from __future__ import annotations
import hashlib,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def cpu_snapshot(obj):
 import torch
 if isinstance(obj,torch.Tensor):return obj.detach().to('cpu',copy=True).contiguous()
 if isinstance(obj,dict):return {k:cpu_snapshot(v) for k,v in obj.items()}
 if isinstance(obj,(list,tuple)):
  items=[cpu_snapshot(x) for x in obj]
  return tuple(items) if isinstance(obj,tuple) else items
 if isinstance(obj,(int,float,str,bool,type(None))):return obj
 raise TypeError('unsafe state type '+str(type(obj)))

def verify_load(path:Path):
 import torch
 mpath=path.with_suffix(path.suffix+'.json')
 meta=json.loads(mpath.read_text())
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest!=meta['sha256']:raise ValueError('checkpoint SHA mismatch')
 state=torch.load(path,map_location='cpu',weights_only=True)
 if state.get('step')!=meta['step']:raise ValueError('checkpoint/manifest step mismatch')
 if not {'trainable','optimizer','data_rng','torch_rng','cuda_rng'}.issubset(state):raise ValueError('incomplete state')
 return state

class AtomicTrainingCheckpoint:
 def __init__(self,folder:Path,async_write:bool):
  self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
  self.async_write=async_write
  self.executor=ThreadPoolExecutor(max_workers=1) if async_write else None
  self.pending=None
  self.closed=False
  self.last_step=0
 def _serialize(self,step,state):
  import torch
  path=self.folder/f'step_{step:06d}.pt'
  tmp=path.with_suffix('.pt.incomplete')
  start=time.perf_counter()
  try:
   torch.save(state,tmp)
   digest=hashlib.sha256(tmp.read_bytes()).hexdigest()
   os.replace(tmp,path)
   meta={'step':step,'sha256':digest,'bytes':path.stat().st_size,'save_worker_ms':(time.perf_counter()-start)*1000}
   tmp_manifest=path.with_suffix('.pt.json.incomplete');manifest=path.with_suffix('.pt.json')
   tmp_manifest.write_text(json.dumps(meta,indent=2)+'\n');os.replace(tmp_manifest,manifest)
   return meta
  finally:tmp.unlink(missing_ok=True)
 def submit(self,step:int,state):
  if self.closed or step<=self.last_step:raise ValueError('writer closed/nonmonotonic step')
  if self.pending:
   if not self.pending.done():raise RuntimeError('bounded pipeline full: cannot silently skip checkpoint')
   self.pending.result()
  t=time.perf_counter();snapshot=cpu_snapshot(state);copy_ms=(time.perf_counter()-t)*1000
  if self.async_write:self.pending=self.executor.submit(self._serialize,step,snapshot);return copy_ms
  self._serialize(step,snapshot);return (time.perf_counter()-t)*1000
 def flush(self):return self.pending.result() if self.pending else None
 def close(self):
  if self.closed:return
  try:self.flush()
  finally:
   self.closed=True
   if self.executor:self.executor.shutdown(wait=True)
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
