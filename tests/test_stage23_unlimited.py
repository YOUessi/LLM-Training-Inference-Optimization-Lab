import unittest
from llm_opt_lab.serving import stage14_mixed_bench
from llm_opt_lab.serving import stage23_unlimited_bench
from llm_opt_lab.serving.stage13_policy import AdmissionGate

class Stage23AdmissionTests(unittest.TestCase):
 def test_unlimited_accepts_even_past_deadline(self):
  gate=AdmissionGate('unlimited')
  d=gate.acquire('label',300,-1000.)
  self.assertTrue(d.accepted)
  self.assertEqual(d.policy,'unlimited')
 def test_import_does_not_mutate_stage14(self):
  self.assertIn('k8s',stage14_mixed_bench.DOMAINS)
  self.assertNotIn('unlimited',stage14_mixed_bench.POLICIES)
