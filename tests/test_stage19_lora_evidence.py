"""CPU-side verification of measured real-Transformer LoRA evidence."""
import hashlib,json,pathlib,unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]/'evidence/stage19_lora'

class LoRAEvidenceTests(unittest.TestCase):
 def test_adapter_hash_shape_and_trainable_only_keys(self):
  from safetensors import safe_open
  for accum in (1,4):
   with self.subTest(accum=accum):
    p=ROOT/f'stage19_qwen15b_lora_v2_accum{accum}'
    payload=json.loads(p.with_suffix('.json').read_text())
    file=p.with_suffix('.safetensors')
    self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),payload['adapter_sha256'])
    self.assertEqual(payload['trainable_parameters'],1089536)
    self.assertEqual(payload['modules_wrapped'],56)
    with safe_open(str(file),framework='pt',device='cpu') as weights:
     keys=list(weights.keys())
     self.assertEqual(len(keys),112)
     self.assertEqual(sum(weights.get_tensor(k).numel() for k in keys),1089536)
     self.assertTrue(all(k.endswith(('.A','.B')) for k in keys))
    self.assertGreater(payload['heldout_loss_before'],payload['heldout_loss_after'])
if __name__=='__main__':unittest.main()
