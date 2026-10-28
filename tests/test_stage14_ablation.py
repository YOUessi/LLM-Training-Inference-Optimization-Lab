"""Factorial mechanism checks; not experimental evidence of throughput improvement."""
import unittest
from llm_opt_lab.serving.stage14_ablation import ABLATIONS, deadline_ms, priority

class Stage14MechanismTests(unittest.TestCase):
    def test_all_variants_have_unique_binary_mechanisms(self):
        self.assertEqual(len(ABLATIONS), 8)
        self.assertEqual(len({(c.dynamic_deadline,c.short_job_first,c.aging,c.online_ewma)
                              for c in ABLATIONS.values()}),8)

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

class Stage14LiveQueueTests(unittest.TestCase):
    def test_all_variants_release_and_account(self):
        import time
        from llm_opt_lab.serving.stage14_ablation import AblationAdmissionGate
        for name in ABLATIONS:
            with self.subTest(name=name):
                gate=AblationAdmissionGate(name)
                decision=gate.acquire('label',400,time.perf_counter())
                self.assertTrue(decision.accepted)
                self.assertEqual(decision.policy,name)
                gate.release('label',800.)
                state=gate.final_state()
                self.assertEqual(state['active'],0)
                self.assertEqual(state['waiters'],0)
                self.assertEqual(state['completed_observations']['label'],int(ABLATIONS[name].online_ewma))

    def test_full_queue_rejects_and_expired_ticket_cleans_up(self):
        import time
        from llm_opt_lab.serving.stage14_ablation import AblationAdmissionGate
        for name in ABLATIONS:
            with self.subTest(name=name):
                gate=AblationAdmissionGate(name)
                for _ in range(8):
                    self.assertTrue(gate.acquire('detailed',400,time.perf_counter()).accepted)
                expired=gate.acquire('detailed',400,time.perf_counter()-2.)
                self.assertEqual(expired.reason,'deadline_expired')
                self.assertEqual(gate.final_state()['waiters'],0)
                for _ in range(8): gate.release('detailed',None)
                self.assertEqual(gate.final_state()['active'],0)

    def test_ewma_factor_is_separate_from_deadline_and_priority(self):
        import time
        from llm_opt_lab.serving.stage14_ablation import AblationAdmissionGate
        gates=[AblationAdmissionGate(n) for n in ('fifo_static400','fifo_static400_ewma')]
        for gate in gates:
            gate.acquire('label',400,time.perf_counter())
            gate.release('label',1000.)
        self.assertEqual(gates[0].final_state()['completed_observations']['label'],0)
        self.assertEqual(gates[1].final_state()['completed_observations']['label'],1)
        self.assertNotEqual(gates[0].final_state()['ewma_service_ms_by_mode']['label'],
                            gates[1].final_state()['ewma_service_ms_by_mode']['label'])

class Stage14DriverTests(unittest.TestCase):
    def test_driver_uses_new_seeds_and_eight_variants(self):
        from llm_opt_lab.serving.stage14_mixed_bench import DOMAINS, POLICIES, parse_args
        self.assertEqual(DOMAINS['k8s'][2],(8101,8109))
        self.assertEqual(DOMAINS['prose'][2],(8201,8209))
        self.assertEqual(set(POLICIES),set(ABLATIONS))
        args=parse_args(['--model','local','--domain','k8s','--seed','8101',
                         '--rate','16','--policy','full_adaptive','--output','/tmp/dev.json'])
        self.assertEqual(args.policy,'full_adaptive')
