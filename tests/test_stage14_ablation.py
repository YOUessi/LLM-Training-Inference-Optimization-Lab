"""Factorial mechanism checks; not experimental evidence of throughput improvement."""
import unittest
from llm_opt_lab.serving.stage14_ablation import ABLATIONS, deadline_ms, priority

class Stage14MechanismTests(unittest.TestCase):
    def test_all_variants_have_unique_binary_mechanisms(self):
        self.assertEqual(len(ABLATIONS), 6)
        self.assertEqual(len({(c.dynamic_deadline,c.short_job_first,c.aging,c.online_ewma)
                              for c in ABLATIONS.values()}),6)

    def test_deadline_factor_independent_of_ordering(self):
        for key in ("fifo_static400","sjf_static400","sjf_aging_static400"):
            self.assertEqual(deadline_ms(ABLATIONS[key],1100,8,4,390),400.)
        self.assertLess(deadline_ms(ABLATIONS["fifo_dynamic"],1100,8,4,390),400.)
        self.assertEqual(deadline_ms(ABLATIONS["fifo_dynamic"],1100,8,4,390),
                         deadline_ms(ABLATIONS["full_adaptive"],1100,8,4,390))

    def test_fifo_sjf_and_aging_are_separable(self):
        old=(1000.,600.,0)
        new=(50.,20.,1)
        self.assertLess(priority(ABLATIONS["fifo_static400"],*old),
                        priority(ABLATIONS["fifo_static400"],*new))
        self.assertGreater(priority(ABLATIONS["sjf_static400"],*old),
                           priority(ABLATIONS["sjf_static400"],*new))
        self.assertLess(priority(ABLATIONS["sjf_aging_static400"],*old),
                        priority(ABLATIONS["sjf_aging_static400"],*new))

    def test_bad_ticket_rejected(self):
        with self.assertRaises(ValueError):
            priority(ABLATIONS["full_adaptive"],-1,0,0)

if __name__=='__main__': unittest.main()
