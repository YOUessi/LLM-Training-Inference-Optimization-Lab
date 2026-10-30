"""Prevent stage40 failure due to wrong replay-kind variable reference."""
import unittest
from llm_opt_lab.training.stage40_constraint_rehearsal import REPLAY_KINDS,choose_replay_kind
class Stage40ReplayKindTests(unittest.TestCase):
 def test_all_512_times_2_deterministic_registered_formats(self):
  self.assertEqual(REPLAY_KINDS,('json_exact','xml_tag','begin_end'))
  for epoch in range(2):
   for i in range(512):
    self.assertEqual(choose_replay_kind(i,epoch),REPLAY_KINDS[(i+epoch)%3])
 def test_out_of_bounds_fails_closed(self):
  for idx,ep in ((-1,0),(512,0),(0,-1),(0,2)):
   with self.assertRaises(ValueError):choose_replay_kind(idx,ep)
