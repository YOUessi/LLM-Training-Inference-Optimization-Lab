import json,tempfile,unittest,torch
from pathlib import Path
from llm_opt_lab.training.stage28_checkpoint import AtomicTrainingCheckpoint,verify_load,cpu_snapshot
class CheckpointContractTests(unittest.TestCase):
 def test_snapshot_immutable_and_restore_optimizer_rng(self):
  with tempfile.TemporaryDirectory() as d:
   x=torch.arange(20,dtype=torch.float32)
   state={'step':3,'trainable':{'layer.A':x},'optimizer':{'state':{0:{'step':torch.tensor(3), 'exp_avg':x.clone()}},'param_groups':[{'lr':1e-3,'params':[0]}]},
       'data_rng':torch.Generator().manual_seed(12).get_state(), 'torch_rng':torch.get_rng_state(),'cuda_rng':[]}
   for use_async in (False,True):
    with AtomicTrainingCheckpoint(Path(d)/str(use_async),async_write=use_async) as w:
     w.submit(3,state)
     w.flush()
    recovered=verify_load(Path(d)/str(use_async)/'step_000003.pt')
    self.assertTrue(torch.equal(recovered['trainable']['layer.A'],torch.arange(20,dtype=torch.float32)))
    self.assertEqual(recovered['step'],3)
 def test_reject_unsafe_type(self):
  with self.assertRaises(TypeError):cpu_snapshot(object())
if __name__=='__main__':unittest.main()
