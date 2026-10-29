"""Verify Qwen3 operator workload parameters and pure-PyTorch reference semantics."""
import json,torch,unittest
class TransformerKernelContract(unittest.TestCase):
 def test_qwen3_dimensions_and_reference(self):
  c=json.load(open('/home/you/models/Qwen3-8B/config.json'))
  self.assertEqual(c['hidden_size'],4096)
  self.assertEqual(c['num_attention_heads'],32)
  self.assertEqual(c['rms_norm_eps'],1e-6)
  x=torch.randn((4,64),dtype=torch.float32)
  r=torch.randn_like(x);g=torch.ones((64,))
  z=x+r;y=(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-6))*g
  self.assertTrue(torch.isfinite(y).all())
 def test_masked_softmax_causality(self):
  x=torch.randn((2,8,8),dtype=torch.float32)
  mask=torch.arange(8).view(1,1,8)>torch.arange(8).view(1,8,1)
  p=torch.softmax(x.masked_fill(mask,-float('inf')),dim=-1)
  self.assertTrue(torch.allclose(p.sum(-1),torch.ones((2,8)),atol=1e-5))
  self.assertEqual(float(p[0,0,1:].abs().sum()),0.)
if __name__=='__main__':unittest.main()
