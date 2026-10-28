"""Atomic bounded asynchronous safe-tensors checkpoint writer.

Only immutable detached CPU snapshots cross into the background thread.
No untrusted pickle/remote code loading. A pending write is a hard backpressure
condition instead of dropping a checkpoint or silently racing its state.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping
import hashlib,json,os,threading

class AsyncCheckpointWriter:
    def __init__(self, root: Path):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='checkpoint')
        self.pending=None
        self.next_step=0
        self.closed=False

    def submit(self, step: int, tensors: Mapping[str,object]) -> None:
        import torch
        if self.closed or step<=self.next_step:raise ValueError('writer closed or non-monotonic step')
        if self.pending is not None:
            if not self.pending.done():raise RuntimeError('checkpoint in progress: backpressure')
            self.pending.result()
        snapshot={name:t.detach().to('cpu',copy=True).contiguous()
                  for name,t in tensors.items()}
        if not snapshot or not all(isinstance(v,torch.Tensor) and torch.isfinite(v).all()
                                   for v in snapshot.values()):raise ValueError('empty or invalid snapshot')
        self.next_step=step
        self.pending=self.executor.submit(self._write,step,snapshot)

    def _write(self,step:int,tensors:dict) -> dict:
        from safetensors.torch import save_file
        filename=f'adapter_step{step:06d}.safetensors'
        target=self.root/filename
        temp=self.root/(filename+'.incomplete')
        try:
            save_file(tensors,str(temp),metadata={'step':str(step),'type':'trainable-adapter-only'})
            digest=hashlib.sha256(temp.read_bytes()).hexdigest()
            os.replace(temp,target)
            evidence={'step':step,'name':filename,'sha256':digest,'trainable_numel':sum(t.numel() for t in tensors.values())}
            manifest=self.root/(filename+'.json')
            scratch=self.root/(filename+'.json.incomplete')
            scratch.write_text(json.dumps(evidence,indent=2)+'\n')
            os.replace(scratch,manifest)
            return evidence
        finally:
            temp.unlink(missing_ok=True)

    def flush(self):
        if self.pending is None:return None
        return self.pending.result()

    def close(self):
        try:return self.flush()
        finally:self.closed=True;self.executor.shutdown(wait=True)

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
