"""Prevent stage41 reading a known-invalid old frozen Stage40 run."""
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Stage41DependencyGateTests(unittest.TestCase):
 def test_runner_uses_only_fixed_source(self):
  source=(ROOT/'scripts/run_stage41_new128.sh').read_text()
  self.assertIn('/home/you/projects/llmopt_stage40_v2_frozen/results/stage40_real_constraint_rehearsal',source)
  self.assertNotIn('/home/you/projects/llmopt_stage40_frozen/results/stage40_real_constraint_rehearsal',source)
  self.assertIn('final_test60_official.json',source)
  self.assertIn('final_untouched101_test.json',source)
