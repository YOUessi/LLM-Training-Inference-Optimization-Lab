"""A workload-aware, age-fair admission queue for controlled LLM experiments.

New system mechanism (client proxy, NOT vLLM engine modification):

1. Estimate service demand using ONLY earlier Stage10 complete-response
   observations: label ~55.49ms, detailed ~1077.63ms.
2. Adjust predicted demand for input Tokens and per-mode EWMA of *real*
   completed model service times, updated online after responses finish.
3. When GPU-serving slots are full, keep at most four local waiters;
   choose the lowest predicted service time plus an aging term
   (short-job-first with bounded starvation via wait-time aging).
4. Compute *per-arrival* admission deadline from predicted work,
   concurrent active count and pending queue depth.
5. Preserve queue-full and deadline-expired rejection reasons.

These are fixed engineering heuristics, NOT a trained performance model.
Do not call this a guaranteed fair or optimal scheduler. Validate against
both lower-complexity static policies and unlimited admission, counting
every offered request as part of the SLO denominator.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

BASE_SERVICE_MS = {"label": 55.49, "detailed": 1077.63}
NOMINAL_PROMPT_TOKENS = 390.
MAX_INFLIGHT = 8
MAX_WAITING = 4
DEADLINE_MIN_MS = 100.
DEADLINE_MAX_MS = 400.
EWMA_ALPHA = .20
AGING_WEIGHT = 2.0  # ms score reduction per ms time spent waiting


def predicted_service_ms(mode: str, prompt_tokens: int,
                         recent_mode_ms: dict[str,float] | None = None) -> float:
    if mode not in BASE_SERVICE_MS:
        raise ValueError("unsupported output mode")
    if not isinstance(prompt_tokens,int) or not 1 <= prompt_tokens <= 2048:
        raise ValueError("invalid calibrated input Tokens")
    base=(recent_mode_ms or BASE_SERVICE_MS)[mode]
    multiplier=.84 + .16*(prompt_tokens/NOMINAL_PROMPT_TOKENS)
    return max(10.,min(4000.,base*multiplier))


def admission_budget_ms(
    estimate_ms: float, *, active: int, waiting: int,
    prompt_tokens: int,
) -> float:
    """Dynamic budget, independent of response timing under evaluation.

    Long costly jobs may wait longer when idle, but deadline tightens
    under congestion. The active/queue pressure is a live state signal.
    """
    if not math.isfinite(estimate_ms) or estimate_ms<=0:
        raise ValueError("invalid workload estimate")
    if not 0 <= active <= MAX_INFLIGHT or not 0 <= waiting <= MAX_WAITING:
        raise ValueError("impossible admission queue occupancy")
    if not 1 <= prompt_tokens <= 2048:
        raise ValueError("invalid prompt Tokens")
    budget=(135. + .20*min(estimate_ms,1200.)
            - 110.*active/MAX_INFLIGHT
            - 65.*waiting/MAX_WAITING
            + .02*(prompt_tokens-390.))
    return round(max(DEADLINE_MIN_MS,min(DEADLINE_MAX_MS,budget)),3)


def fair_priority(estimated_ms:float, waited_ms:float, order:int) -> tuple[float,int]:
    if estimated_ms<=0 or waited_ms<0:
        raise ValueError("invalid demand/queue waiting age")
    return (estimated_ms-AGING_WEIGHT*waited_ms,order)


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str | None
    deadline_ms: float
    predicted_service_ms: float
    active_on_arrival: int
    waiting_on_arrival: int
    policy: str

    def as_dict(self) -> dict[str,Any]:
        return {
            "accepted":self.accepted,
            "rejection_reason":self.reason,
            "dynamic_admission_deadline_ms":self.deadline_ms,
            "predicted_service_ms":round(self.predicted_service_ms,3),
            "active_at_arrival":self.active_on_arrival,
            "waiting_at_arrival":self.waiting_on_arrival,
            "policy_name":self.policy,
        }


class AdmissionGate:
    def __init__(self,policy:str):
        if policy not in ("unlimited","fixed150","fixed400","adaptive"):
            raise ValueError("unapproved admission policy")
        self.policy=policy
        self.cv=threading.Condition()
        self.active=0
        self.waiters: list[dict[str,Any]]=[]
        self.ewma=BASE_SERVICE_MS.copy()
        self.counter=0
        self.recorded_completion_count={"label":0,"detailed":0}

    def acquire(self,mode:str,prompt_tokens:int,arrival_due:float) -> Decision:
        if self.policy=="unlimited":
            return Decision(True,None,0.,predicted_service_ms(mode,prompt_tokens),
                            0,0,"unlimited")

        with self.cv:
            estimate=predicted_service_ms(mode,prompt_tokens,self.ewma)
            active=self.active
            waiting=len(self.waiters)
            if self.policy=="adaptive":
                deadline=admission_budget_ms(
                    estimate,active=active,waiting=waiting,
                    prompt_tokens=prompt_tokens)
            else:
                deadline=150. if self.policy=="fixed150" else 400.
            common={
                "deadline_ms":deadline,
                "predicted_service_ms":estimate,
                "active_on_arrival":active,
                "waiting_on_arrival":waiting,
                "policy":self.policy,
            }
            # Keep the previous Stage10 fixed baseline: immediate free
            # capacity is accepted even if the nominal waiting deadline
            # has already elapsed. The deadline only limits queue wait.
            if self.active<MAX_INFLIGHT and not self.waiters:
                self.active+=1
                return Decision(True,None,**common)
            if len(self.waiters)>=MAX_WAITING:
                return Decision(False,"queue_full",**common)

            order=self.counter
            self.counter+=1
            ticket={
                "order":order,"predicted":estimate,
                "joined":time.perf_counter(),
                "deadline":arrival_due+(deadline/1000.),
            }
            self.waiters.append(ticket)
            while True:
                now=time.perf_counter()
                left=ticket["deadline"]-now
                if left<=0:
                    self.waiters.remove(ticket)
                    self.cv.notify_all()
                    return Decision(False,"deadline_expired",**common)
                if self.active<MAX_INFLIGHT:
                    # For fixed policies use FIFO order; for adaptive use
                    # shortest estimated service with waiting-time aging.
                    if self.policy=="adaptive":
                        candidate=min(
                            (x for x in self.waiters if x["deadline"]>now),
                            key=lambda x: fair_priority(
                                x["predicted"],(now-x["joined"])*1000.,x["order"]),
                            default=None,
                        )
                    else:
                        candidate=min(self.waiters,key=lambda x:x["order"])
                    if candidate is ticket:
                        self.waiters.remove(ticket)
                        self.active+=1
                        self.cv.notify_all()
                        return Decision(True,None,**common)
                self.cv.wait(timeout=min(left,.10))

    def release(self,mode:str,real_service_latency_ms:float | None) -> None:
        if self.policy=="unlimited":
            return
        with self.cv:
            if self.active<=0:
                raise RuntimeError("gate released without an acquired slot")
            self.active-=1
            if (self.policy=="adaptive" and mode in self.ewma
                and real_service_latency_ms is not None
                and math.isfinite(real_service_latency_ms)
                and real_service_latency_ms>=0):
                old=self.ewma[mode]
                self.ewma[mode]=(1.-EWMA_ALPHA)*old+EWMA_ALPHA*min(
                    4000.,max(10.,real_service_latency_ms)
                )
                self.recorded_completion_count[mode]+=1
            self.cv.notify_all()

    def final_state(self) -> dict[str,Any]:
        with self.cv:
            return {
                "policy":self.policy,
                "active":self.active,
                "waiters":len(self.waiters),
                "ewma_service_ms_by_mode":{k:round(v,3) for k,v in self.ewma.items()},
                "completed_observations":self.recorded_completion_count.copy(),
            }
