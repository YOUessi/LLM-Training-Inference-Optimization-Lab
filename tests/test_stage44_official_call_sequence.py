"""Reproduce actual IFBench N=None kwargs failure; official two-pass sequence."""
import unittest
from evaluation_lib import InputExample,test_instruction_following_strict,test_instruction_following_loose
class OfficialIFBenchCallSequenceTests(unittest.TestCase):
 def test_original_N_none_requires_same_mutated_input(self):
  def sample():
   return InputExample(key='dummy',instruction_id_list=['ratio:sentence_type'],
      prompt='Write a short sentence.',kwargs=[{'N':None}])
  p=sample()
  first=test_instruction_following_strict(p,{p.prompt:'A short sentence.'})
  second=test_instruction_following_loose(p,{p.prompt:'A short sentence.'})
  self.assertEqual(len(first.follow_instruction_list),1)
  self.assertEqual(len(second.follow_instruction_list),1)
  with self.assertRaises(TypeError):
   other=sample()
   test_instruction_following_loose(other,{other.prompt:'A short sentence.'})
