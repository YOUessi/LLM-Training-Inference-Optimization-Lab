import hashlib,json,tempfile,unittest
from pathlib import Path

from llm_opt_lab.training.async_checkpoint import AsyncCheckpointWriter

class AsyncCheckpointTests(unittest.TestCase):
 def test_atomic_immutable_checkpoint_and_sha(self):
  import torch
  from safetensors.torch import load_file
  with tempfile.TemporaryDirectory() as directory:
   source=torch.ones((16,20),dtype=torch.float32)
   writer=AsyncCheckpointWriter(Path(directory))
   writer.submit(1,{'layer.A':source})
   source.zero_()
   evidence=writer.flush()
   data=Path(directory)/evidence['name']
   self.assertEqual(evidence['sha256'],hashlib.sha256(data.read_bytes()).hexdigest())
   self.assertTrue(torch.equal(load_file(str(data))['layer.A'],torch.ones_like(source)))
   self.assertEqual(json.loads((Path(directory)/(evidence['name']+'.json')).read_text())['step'],1)
   self.assertEqual(list(Path(directory).glob('*.incomplete')),[])
   with self.assertRaises(ValueError):writer.submit(1,{'layer.A':source})
   writer.close()
 def test_empty_invalid_state_rejected(self):
  import torch
  with tempfile.TemporaryDirectory() as directory:
   with AsyncCheckpointWriter(Path(directory)) as writer:
    with self.assertRaises(ValueError):writer.submit(1,{})
    with self.assertRaises(ValueError):writer.submit(2,{'bad':torch.tensor([float('nan')])})
