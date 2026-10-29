import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'evidence/stage27_measured'
class Stage27EvidenceTests(unittest.TestCase):
 def test_all_8_gpu_cases(self):
  d=json.loads((ROOT/'benchmark.json').read_text())
  self.assertEqual(d['status'],'measured');self.assertEqual(len(d['cases']),8)
  for c in d['cases']:
   self.assertTrue(c['numerical_max_abs_error']<=c['atol'])
   self.assertGreater(c['timings']['torch_compile']['median_ms'],0)
   self.assertGreater(c['timings']['custom_triton']['median_ms'],0)
   self.assertEqual(len(c['timings']['custom_triton']['repeats_ms']),3)
  sha=(ROOT/'SHA256SUMS').read_text().split()[0]
  self.assertEqual(sha,hashlib.sha256((ROOT/'benchmark.json').read_bytes()).hexdigest())
  self.assertEqual(len(d['profiler']),2)
