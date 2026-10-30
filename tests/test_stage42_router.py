"""No benchmark labels/keys in route; stable request-only behavior."""
import unittest
from llm_opt_lab.training.stage42_risk_router import choose_model,PATTERNS,THRESHOLD
class RouterContractTests(unittest.TestCase):
 def test_basic_instruction_risk(self):
  for user in ('Please output exactly 3 sentences.',
               'Return JSON with exactly an answer key.',
               'Do not use any commas in your answer.',
               'Start with the word OK and end with DONE.',
               'Respond only in Spanish.',
               'Write bullet points for each step.'):
   self.assertEqual(choose_model(user).choice,'base',user)
 def test_normal_task_uses_trainable_specialist(self):
  for user in ('How does an aircraft fly?',
               'Why are neural networks good at recognizing images?',
               'What happens when you heat water?',
               'Tell me about the history of the internet.'):
   self.assertEqual(choose_model(user).choice,'lora_sft',user)
 def test_rules_have_no_references_to_test_answer_or_id(self):
  source=' '.join(p[1] for p in PATTERNS).lower()
  self.assertNotIn('ifeval',source)
  self.assertNotIn('stage42',source)
  self.assertEqual(THRESHOLD,2)
  self.assertEqual(choose_model('What is the meaning of life?'),choose_model('What is the meaning of life?'))
 def test_empty_or_extremely_long_prompt_rejected(self):
  for text in ('','   ','a'*25001):
   with self.assertRaises(ValueError):choose_model(text)
