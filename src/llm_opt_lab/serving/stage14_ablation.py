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
    "sjf_static400": SchedulerAblation("sjf_static400", False, True, False, True),
    "sjf_aging_static400": SchedulerAblation("sjf_aging_static400", False, True, True, True),
    "fifo_dynamic": SchedulerAblation("fifo_dynamic", True, False, False, True),
    "sjf_dynamic": SchedulerAblation("sjf_dynamic", True, True, False, True),
    "full_adaptive": SchedulerAblation("full_adaptive", True, True, True, True),
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
