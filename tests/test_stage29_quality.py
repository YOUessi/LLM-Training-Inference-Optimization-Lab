import unittest
from llm_opt_lab.training.stage29_instruction_quality import quality,tokenize
class Stage29QualityTests(unittest.TestCase):
 def test_exact_rouge_and_lowercase(self):
  x=quality('Alpha beta','alpha BETA')
  self.assertEqual(x['normalized_exact_match'],1)
  self.assertEqual(x['rouge_l_f1'],1.)
  self.assertEqual(x['token_f1'],1.)
 def test_wrong_word_order_is_not_exact(self):
  x=quality('alpha beta gamma','gamma beta alpha')
  self.assertEqual(x['normalized_exact_match'],0)
  self.assertLess(x['rouge_l_f1'],1.)
  self.assertEqual(x['token_f1'],1.)
 def test_empty_prediction_handled(self):
  x=quality('hello world','')
  self.assertEqual(x['token_f1'],0.)
  self.assertEqual(x['rouge_l_f1'],0.)
if __name__=='__main__':unittest.main()
