"""Stage14 independently switchable client-proxy scheduler mechanisms.

This module is CPU-testable; no GPU results or efficacy claims are implied.
Stage13's original implementation is intentionally left frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from llm_opt_lab.serving.stage13_policy import admission_budget_ms, fair_priority

@dataclass(frozen=True)
class SchedulerAblation:
    name: str
    dynamic_deadline: bool
    short_job_first: bool
    aging: bool
    online_ewma: bool

ABLATIONS = {
    "fifo_static400": SchedulerAblation("fifo_static400", False, False, False, False),
    "fifo_static400_ewma": SchedulerAblation("fifo_static400_ewma", False, False, False, True),
    "sjf_static400_noewma": SchedulerAblation("sjf_static400_noewma", False, True, False, False),
    "sjf_static400": SchedulerAblation("sjf_static400", False, True, False, True),
    "sjf_aging_static400": SchedulerAblation("sjf_aging_static400", False, True, True, True),
    "fifo_dynamic": SchedulerAblation("fifo_dynamic", True, False, False, True),
    "sjf_dynamic": SchedulerAblation("sjf_dynamic", True, True, False, True),
    "full_adaptive": SchedulerAblation("full_adaptive", True, True, True, True),
    "full_8b_calibrated": SchedulerAblation("full_8b_calibrated", True, True, True, True),
}

def deadline_ms(config: SchedulerAblation, estimated_ms: float, active: int,
                waiting: int, prompt_tokens: int) -> float:
    return (admission_budget_ms(estimated_ms, active=active, waiting=waiting,
                                prompt_tokens=prompt_tokens)
            if config.dynamic_deadline else 400.)

def priority(config: SchedulerAblation, estimated_ms: float,
             waited_ms: float, order: int) -> tuple[float, int]:
    if estimated_ms <= 0 or waited_ms < 0 or order < 0:
        raise ValueError("invalid scheduling ticket")
    if not config.short_job_first:
        return (0., order)
    if config.aging:
        return fair_priority(estimated_ms, waited_ms, order)
    return (estimated_ms, order)

# Separate live queue implementation: never alter archived Stage13 source.
import math
import threading
import time
from llm_opt_lab.serving.stage13_policy import (
    BASE_SERVICE_MS, MAX_INFLIGHT, MAX_WAITING, EWMA_ALPHA,
    Decision, predicted_service_ms,
)

class AblationAdmissionGate:
    def __init__(self, variant: str):
        if variant not in ABLATIONS:
            raise ValueError("unregistered Stage14 variant")
        self.config = ABLATIONS[variant]
        self.cv = threading.Condition()
        self.active = 0
        self.waiters = []
        # Frozen cross-domain diagnostic constants, not GPU service timings.
        self.ewma = ({"label":323.245753,"detailed":2287.51214}
            if variant=="full_8b_calibrated" else BASE_SERVICE_MS.copy())
        self.completed = {"label":0, "detailed":0}
        self.order = 0

    def acquire(self, mode: str, prompt_tokens: int, arrival_due: float) -> Decision:
        with self.cv:
            estimate = predicted_service_ms(mode, prompt_tokens,
                self.ewma if self.config.online_ewma else BASE_SERVICE_MS)
            active, waiting = self.active, len(self.waiters)
            budget = deadline_ms(self.config,estimate,active,waiting,prompt_tokens)
            args = dict(deadline_ms=budget,predicted_service_ms=estimate,
                        active_on_arrival=active,waiting_on_arrival=waiting,
                        policy=self.config.name)
            if active < MAX_INFLIGHT and not self.waiters:
                self.active += 1
                return Decision(True,None,**args)
            if waiting >= MAX_WAITING:
                return Decision(False,"queue_full",**args)
            ticket = dict(order=self.order,predicted=estimate,
                          joined=time.perf_counter(),deadline=arrival_due+budget/1000.)
            self.order += 1
            self.waiters.append(ticket)
            while True:
                now = time.perf_counter()
                if now >= ticket["deadline"]:
                    self.waiters.remove(ticket)
                    self.cv.notify_all()
                    return Decision(False,"deadline_expired",**args)
                if self.active < MAX_INFLIGHT:
                    eligible=[t for t in self.waiters if t["deadline"]>now]
                    candidate=min(eligible,key=lambda t: priority(
                        self.config,t["predicted"],(now-t["joined"])*1000.,t["order"]),
                        default=None)
                    if candidate is ticket:
                        self.waiters.remove(ticket)
                        self.active += 1
                        self.cv.notify_all()
                        return Decision(True,None,**args)
                self.cv.wait(timeout=min(.10,ticket["deadline"]-now))

    def release(self, mode: str, real_service_latency_ms: float | None) -> None:
        with self.cv:
            if self.active <= 0:
                raise RuntimeError("release without acquired slot")
            self.active -= 1
            if (self.config.online_ewma and mode in self.ewma
                and real_service_latency_ms is not None
                and math.isfinite(real_service_latency_ms)
                and real_service_latency_ms >= 0):
                self.ewma[mode] = ((1.-EWMA_ALPHA)*self.ewma[mode] +
                    EWMA_ALPHA*min(4000.,max(10.,real_service_latency_ms)))
                self.completed[mode] += 1
            self.cv.notify_all()

    def final_state(self) -> dict:
        with self.cv:
            return dict(policy=self.config.name,active=self.active,
                waiters=len(self.waiters),ewma_service_ms_by_mode=self.ewma.copy(),
                completed_observations=self.completed.copy())
