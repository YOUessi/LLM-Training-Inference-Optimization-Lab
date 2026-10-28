"""Stage13 dynamic admission policy correctness and non-guaranteed fairness contract."""
from __future__ import annotations

import threading
import time
import unittest

from llm_opt_lab.serving.stage13_policy import (
    AdmissionGate, BASE_SERVICE_MS, MAX_INFLIGHT, MAX_WAITING,
    admission_budget_ms, fair_priority, predicted_service_ms,
)


class Stage13AdmissionPolicyTests(unittest.TestCase):
    def test_workload_prediction_uses_mode_and_real_input_token_count(self):
        brief=predicted_service_ms("label",400)
        long=predicted_service_ms("detailed",400)
        self.assertLess(brief,90)
        self.assertGreater(long,900)
        self.assertGreater(predicted_service_ms("label",680),
                           predicted_service_ms("label",170))
        with self.assertRaises(ValueError):
            predicted_service_ms("invalid",200)
        with self.assertRaises(ValueError):
            predicted_service_ms("label",5000)

    def test_dynamic_budget_responds_to_predicted_work_and_queue(self):
        small=admission_budget_ms(60.,active=0,waiting=0,prompt_tokens=300)
        large=admission_budget_ms(1100.,active=0,waiting=0,prompt_tokens=300)
        pressured=admission_budget_ms(1100.,active=8,waiting=4,prompt_tokens=300)
        self.assertLess(small,large)
        self.assertLess(pressured,large)
        self.assertLessEqual(pressured,400.)
        self.assertGreaterEqual(pressured,100.)
        with self.assertRaises(ValueError):
            admission_budget_ms(100.,active=9,waiting=0,prompt_tokens=300)

    def test_wait_aging_can_override_short_job_priority(self):
        self.assertLess(fair_priority(50.,20.,1),fair_priority(1000.,20.,2))
        self.assertLess(fair_priority(1000.,600.,2),fair_priority(50.,20.,1))

    def test_bound_capacity_and_deadline_expiry(self):
        gate=AdmissionGate("adaptive")
        decisions=[gate.acquire("detailed",470,time.perf_counter())
                   for _ in range(MAX_INFLIGHT)]
        self.assertEqual(len(decisions),8)
        self.assertTrue(all(d.accepted for d in decisions))
        # With all slots occupied and a due time already in the past,
        # expire deterministically instead of sleeping for a real model.
        expired=gate.acquire("detailed",470,time.perf_counter()-1.)
        self.assertFalse(expired.accepted)
        self.assertEqual(expired.reason,"deadline_expired")
        self.assertEqual(gate.final_state()["active"],8)
        self.assertEqual(gate.final_state()["waiters"],0)
        for _ in range(8):
            gate.release("detailed",1000.)
        self.assertEqual(gate.final_state()["active"],0)
        self.assertEqual(gate.final_state()["waiters"],0)

    def test_release_updates_online_ewma_only_after_real_model_latency(self):
        gate=AdmissionGate("adaptive")
        original=gate.final_state()["ewma_service_ms_by_mode"]["detailed"]
        self.assertTrue(gate.acquire("detailed",400,time.perf_counter()).accepted)
        gate.release("detailed",1800.)
        state=gate.final_state()
        self.assertGreater(state["ewma_service_ms_by_mode"]["detailed"],original)
        self.assertEqual(state["completed_observations"]["detailed"],1)
        self.assertEqual(state["completed_observations"]["label"],0)

    def test_fixed_policy_does_not_train_ewma_from_results(self):
        gate=AdmissionGate("fixed150")
        self.assertTrue(gate.acquire("detailed",400,time.perf_counter()).accepted)
        gate.release("detailed",3500.)
        self.assertEqual(gate.final_state()["completed_observations"]["detailed"],0)
        self.assertAlmostEqual(gate.final_state()["ewma_service_ms_by_mode"]["detailed"],
                               BASE_SERVICE_MS["detailed"],places=2)

    def test_queue_limit_rejects_burst_without_silent_drops(self):
        gate=AdmissionGate("adaptive")
        for _ in range(MAX_INFLIGHT):
            self.assertTrue(gate.acquire("detailed",470,time.perf_counter()).accepted)
        barrier=threading.Barrier(MAX_WAITING+1)
        decisions=[]
        lock=threading.Lock()

        def wait_for_slot():
            barrier.wait()
            r=gate.acquire("detailed",470,time.perf_counter())
            with lock:
                decisions.append(r)
            if r.accepted:
                gate.release("detailed",500.)

        threads=[threading.Thread(target=wait_for_slot) for _ in range(MAX_WAITING)]
        for th in threads:th.start()
        barrier.wait()
        deadline=time.monotonic()+.5
        while gate.final_state()["waiters"]<MAX_WAITING and time.monotonic()<deadline:
            time.sleep(.001)
        self.assertEqual(gate.final_state()["waiters"],MAX_WAITING)
        overflow=gate.acquire("detailed",470,time.perf_counter())
        self.assertFalse(overflow.accepted)
        self.assertEqual(overflow.reason,"queue_full")
        for _ in range(MAX_INFLIGHT):
            gate.release("detailed",1000.)
        for th in threads:th.join(timeout=2.)
        self.assertTrue(all(not th.is_alive() for th in threads))
        self.assertEqual(len(decisions),4)
        self.assertEqual(gate.final_state()["active"],0)
        self.assertEqual(gate.final_state()["waiters"],0)

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            AdmissionGate("arbitrary_python_code")


if __name__ == "__main__":
    unittest.main()
