import unittest
from llm_opt_lab.serving.stage33_sustained_bench import MIXES,RATE,SEEDS
from llm_opt_lab.serving.stage8_arrivals import schedule
class Stage33Contracts(unittest.TestCase):
 def test_offer_window_and_seeds(self):
  self.assertEqual(RATE,1.6)
  self.assertEqual(MIXES,('cold0','hot50'))
  self.assertEqual(SEEDS,(3011,3029,3041))
  for seed in SEEDS:
   offsets=schedule(seed,RATE,96,kind='fixed_window')
   self.assertEqual(offsets[-1],59.375)
