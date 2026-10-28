"""Real open-loop vLLM arrival-rate experiment with multilingual input.

This measures the effect of independent Poisson interarrival schedules and
output instructions using a frozen 24-request, 3-language, 4-length corpus.

Do not treat capped output as actual generation length. Do not conflate client
executor queueing and server-side inference queueing. GPU KV utilization is
sampled opportunistically from real vLLM Prometheus; unavailable != zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_opt_lab.provenance import environment, write_json_atomic
from llm_opt_lab.serving.bench_client import _endpoint, _request, quantile, summarize

RATES = (2.0, 8.0)
SEEDS = (42, 137)
MODES = ("label", "summary", "detailed")
OUTPUT_CAP = 128
CONTEXT_TOLERANCE = 95
MAX_CLIENT_WORKERS = 8
SLO_MS = (100., 250., 500., 1000.)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8026")
    p.add_argument("--model", required=True)
    p.add_argument("--requests", type=Path, default=Path("data/stage8_frozen_requests.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("data/stage8_frozen_requests_manifest.json"))
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--rate", type=float, required=True, help="Poisson target requests per second")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--schedule-kind", choices=("poisson", "fixed_window"), default="poisson",
                   help="fixed_window preserves randomized gaps but normalizes exact sample-average offered rate")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.rate not in RATES:
        p.error("rate must be one of 2 or 8 requests/second")
    if a.seed not in SEEDS:
        p.error("seed must be 42 or 137")
    if a.warmup not in (0, 1):
        p.error("warmup must be 0 or 1")
    return a


def frozen_requests(path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
    b = path.read_bytes()
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(b).hexdigest() != m["request_jsonl_sha256"]:
        raise ValueError("frozen request body checksum mismatch")
    requests = [json.loads(s) for s in b.decode("utf-8").splitlines()]
    if len(requests) != 24 or len({x["request_id"] for x in requests}) != 24:
        raise ValueError("missing or duplicate frozen requests")
    if sorted(m["generation_instructions"]) != sorted(MODES):
        raise ValueError("missing generation mode in frozen instruction manifest")
    return requests, m


def schedule(seed: int, rate: float, n: int, *, kind: str = "poisson") -> list[float]:
    if n <= 1 or rate <= 0 or not math.isfinite(rate):
        raise ValueError("invalid stochastic request arrival specification")
    if kind not in ("poisson", "fixed_window"):
        raise ValueError("unknown arrival schedule kind")
    rng = random.Random(seed)
    offsets = [0.]
    for _ in range(n-1):
        offsets.append(offsets[-1] + rng.expovariate(rate))
    if kind == "fixed_window":
        # Condition the total span while retaining randomized relative gaps.
        # This is NOT a homogeneous Poisson process after normalization.
        span = (n - 1) / rate
        scale = span / offsets[-1]
        offsets = [x * scale for x in offsets]
        offsets[-1] = span
    return offsets


def read_kv_cache_utilization(text: str) -> float | None:
    """Read a real vLLM Prometheus gauge; no fabricated fallback."""
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if name not in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            continue
        try:
            n = float(line.rsplit(None, 1)[-1])
        except ValueError:
            continue
        if math.isfinite(n) and 0 <= n <= 1:
            return n
    return None


def sample_kv_metrics(base_url: str, stop: threading.Event, samples: list[dict]) -> None:
    url = base_url.rstrip("/") + "/metrics"
    while not stop.is_set():
        ts = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                val = read_kv_cache_utilization(r.read().decode("utf-8"))
                if val is not None:
                    samples.append({"offset_monotonic": ts, "kv_usage": val})
        except (OSError, UnicodeError, ValueError):
            pass
        stop.wait(.15)


def group_language(records: list[dict]) -> dict[str, dict]:
    result = {}
    for lang in ("en", "zh-cn", "ja"):
        rows = [x for x in records if x["language"] == lang]
        ok = [x for x in rows if x["status"] == "ok" and x.get("ttft_from_due_ms") is not None]
        result[lang] = {
            "sent": len(rows), "succeeded": len(ok),
            "p95_arrival_to_first_token_ms": quantile([x["ttft_from_due_ms"] for x in ok], .95),
            "p95_service_ttft_ms": quantile([x["ttft_ms"] for x in ok], .95),
            "mean_real_input_tokens": statistics.mean(x["prompt_tokens"] for x in ok) if ok else None,
        }
    return result


def output_length_checks(mode: str, records: list[dict]) -> dict:
    values = [x["completion_tokens"] for x in records if x["status"] == "ok"]
    if not values or any(v is None for v in values):
        return {"status": "fail_missing_output_tokens", "realized_mean": None}
    fraction_early = sum(v < OUTPUT_CAP for v in values) / len(values)
    fraction_very_early = sum(v <= 32 for v in values) / len(values)
    mean = statistics.mean(values)
    checks = {
        "label": fraction_very_early >= .75,
        "summary": fraction_early >= .50,
        "detailed": mean >= 80,
    }
    return {
        "status": "pass" if checks[mode] else "fail_realized_length_gate",
        "realized_mean": mean,
        "realized_min": min(values), "realized_max": max(values),
        "fraction_stopped_before_cap": fraction_early,
        "fraction_32_tokens_or_less": fraction_very_early,
        "mode_gate_satisfied": checks[mode],
    }


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        endpoint = _endpoint(a.base_url)
        items, manifest = frozen_requests(a.requests, a.manifest)
        seed = a.seed
        random.Random(seed).shuffle(items)
        instruction = manifest["generation_instructions"][a.mode]
        if a.warmup:
            candidate = items[-1]
            warm = _request(
                -1, endpoint=endpoint, model=a.model,
                prompt=candidate["prompt_text"], max_tokens=OUTPUT_CAP,
                timeout=90., seed=seed, response_instruction=instruction,
            )
            if warm["status"] != "ok":
                print("BLOCKED: real model warmup failed", file=sys.stderr)
                return 2

        offsets = schedule(seed, a.rate, len(items), kind=a.schedule_kind)
        records: list[dict] = []
        kv_samples: list[dict] = []
        stop = threading.Event()
        sampler = threading.Thread(
            target=sample_kv_metrics,
            args=(a.base_url, stop, kv_samples), daemon=True,
        )
        sampler.start()
        start = time.perf_counter() + .1

        def request_item(i: int, item: dict, due_at: float):
            actual_send = time.perf_counter()
            result = _request(
                i, endpoint=endpoint, model=a.model,
                prompt=item["prompt_text"], max_tokens=OUTPUT_CAP,
                timeout=120., seed=seed, response_instruction=instruction,
            )
            actual_submit_delay = max(0., actual_send - due_at) * 1000.
            result.update({
                "request_id": item["request_id"],
                "language": item["language"], "topic": item["topic"],
                "bucket": item["bucket"],
                "prompt_sha256": item["prompt_sha256"],
                "target_prompt_tokens": item["target_prompt_tokens"],
                "calibrated_prompt_tokens": item["calibrated_prompt_tokens"][a.mode],
                "scheduled_offset_s": due_at - start,
                "client_executor_wait_ms": actual_submit_delay,
                "ttft_from_due_ms": (
                    actual_submit_delay + result["ttft_ms"]
                    if result["ttft_ms"] is not None else None
                ),
                "latency_from_due_ms": actual_submit_delay + result["latency_ms"],
            })
            return result

        try:
            with ThreadPoolExecutor(max_workers=MAX_CLIENT_WORKERS) as pool:
                futures = []
                for i, (offset, item) in enumerate(zip(offsets, items)):
                    due = start + offset
                    time.sleep(max(0., due - time.perf_counter()))
                    futures.append(pool.submit(request_item, i, item, due))
                records = sorted(
                    (f.result() for f in as_completed(futures)),
                    key=lambda x: x["index"],
                )
        finally:
            stop.set()
            sampler.join(timeout=3.)

        end = time.perf_counter()
        elapsed = end-start
        summary = summarize(records, elapsed)
        lengths = output_length_checks(a.mode, records)
        valid_length = all(
            x["status"] == "ok" and isinstance(x.get("prompt_tokens"), int)
            and abs(x["prompt_tokens"]-x["target_prompt_tokens"]) <= CONTEXT_TOLERANCE
            for x in records
        )
        normalized_arrival_rate = (len(offsets)-1) / offsets[-1]
        fixed_rate_valid = (
            a.schedule_kind != "fixed_window"
            or abs(normalized_arrival_rate - a.rate) < 1e-8
        )
        valid = (
            fixed_rate_valid and len(records) == 24 and summary["succeeded"] == 24
            and summary["failed"] == 0 and summary["usage_coverage"] == 1.
            and summary["prompt_usage_coverage"] == 1.
            and valid_length and lengths["status"] == "pass"
        )
        from_due = [x["ttft_from_due_ms"] for x in records if x["ttft_from_due_ms"] is not None]
        actual_length = [x["prompt_tokens"] for x in records if isinstance(x["prompt_tokens"], int)]
        utilization = [x["kv_usage"] for x in kv_samples]
        result = {
            "status": "measured" if valid else "invalid_or_blocked",
            "kind": "real_vllm_stage8_multilingual_openloop",
            "not_model_training": True,
            "environment": environment(),
            "config": {
                "model": a.model, "rate_per_second": a.rate,
                "seed": a.seed, "mode": a.mode, "requests": 24,
                "max_client_workers": MAX_CLIENT_WORKERS,
                "output_cap_tokens": OUTPUT_CAP,
                "warmup_requests": a.warmup,
                "frozen_request_sha256": manifest["request_jsonl_sha256"],
                "source_corpus_sha256": manifest["original_corpus_sha256"],
                "response_instruction_sha256": hashlib.sha256(
                    instruction.encode("utf-8")
                ).hexdigest(),
                "schedule": (
                    "seeded_fixed_duration_exponential_spacings"
                    if a.schedule_kind == "fixed_window"
                    else "seeded_expovariate_open_loop"
                ),
                "schedule_kind": a.schedule_kind,
                "fixed_duration_not_an_unconditioned_poisson_process": a.schedule_kind == "fixed_window",
                "request_arrival_schedule_s": offsets,
                "cache_expected": "explicitly_disabled_in_launcher",
            },
            "summary": summary,
            "length_gates": lengths,
            "all_actual_input_tokens_in_bounds": valid_length,
            "actual_input_token_min": min(actual_length) if actual_length else None,
            "actual_input_token_max": max(actual_length) if actual_length else None,
            "offered_arrival_rate_empirical_per_second": ((len(offsets)-1) / offsets[-1]) if offsets[-1]>0 else None,
            "completed_requests_per_second": len(records) / elapsed if elapsed>0 else None,
            "p95_ttft_from_scheduled_arrival_ms": quantile(from_due, .95),
            "p95_executor_queue_wait_ms": quantile([x["client_executor_wait_ms"] for x in records], .95),
            "ttft_slo_violations": {
                str(int(limit)): sum(
                    x.get("ttft_from_due_ms") is None or x["ttft_from_due_ms"] > limit
                    for x in records
                )
                for limit in SLO_MS
            },
            "by_language": group_language(records),
            "gpu_kv_cache_gauge": {
                "available": bool(utilization),
                "samples": len(utilization),
                "peak_fraction": max(utilization) if utilization else None,
                "average_fraction": statistics.mean(utilization) if utilization else None,
                "source": "vllm Prometheus, periodic 150ms sampling (not a kernel counter)",
            },
            "requests": records,
            "limitations": [
                "Client executor wait and server TTFT are distinct; arrival-to-first-token includes both",
                "Only one GPU and Kubernetes technical docs; not cross-industry representative",
                "Short/long output length varies by requested instruction; not a model quality metric",
                "KV utilization gauge may be unavailable; no zero substitution",
                "Seeded Poisson arrivals are an open-loop synthetic arrival process, not actual production logs",
            ],
        }
        write_json_atomic(a.output, result)
        print(
            f"STAGE8 {a.mode} seed={seed} rate={a.rate}/s "
            f"ok={summary['succeeded']}/24 status={result['status']} "
            f"actual_out={lengths['realized_mean']:.1f} "
            f"p95_service_ttft={summary['p95_ttft_ms']:.1f}ms "
            f"p95_arrival_ttft={result['p95_ttft_from_scheduled_arrival_ms']:.1f}ms "
            f"output_tps={summary['output_tokens_per_second']:.1f}", flush=True
        )
        return 0 if valid else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"STAGE8 INVALID: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
