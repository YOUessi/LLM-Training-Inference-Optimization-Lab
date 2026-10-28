"""CPU-only checks on real 8B NF4 QLoRA adapter evidence and configuration."""
import json,hashlib,unittest
from pathlib import Path
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]/'evidence/stage20_qlora'
class QLoRAEvidenceTests(unittest.TestCase):
 def test_two_full_8b_adapter_artifacts(self):
  for accum in (1,2):
   with self.subTest(accum=accum):
    folder=ROOT/f'accum{accum}'
    metric=json.loads((folder/'metrics.json').read_text())
    self.assertEqual(metric['status'],'measured')
    self.assertEqual(metric['base_model'],'Qwen3-8B')
    self.assertEqual(metric['quantization'],'bitsandbytes-nf4-double-quant')
    self.assertEqual(metric['accum'],accum)
    self.assertEqual(metric['trainable_parameters'],3833856)
    for filename,digest in metric['adapter_sha256'].items():
     self.assertEqual(hashlib.sha256((folder/filename).read_bytes()).hexdigest(),digest)
    with safe_open(str(folder/'adapter_model.safetensors'),framework='pt',device='cpu') as weights:
     keys=weights.keys()
     self.assertEqual(len(keys),144)
     self.assertEqual(sum(weights.get_tensor(k).numel() for k in keys),3833856)
     self.assertTrue(all('lora_A' in k or 'lora_B' in k for k in keys))
 def test_reloaded_adapter_has_observed_posttrain_loss(self):
  data=json.loads((ROOT/'reload_validation.json').read_text())
  self.assertEqual(data['status'],'pass')
  self.assertLess(data['absolute_loss_diff'],0.05)
