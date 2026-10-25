"""OpenAI-compatible streamed LLM serving benchmark.

No third-party dependencies. Starts no server; points at an existing vLLM
endpoint. Requests streaming token usage; unknown TPOT remains null.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from llm_opt_lab.provenance import environment, write_json_atomic
from llm_opt_lab.serving.sse import chunk_text, completion_tokens, iter_sse_json


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if not 0 <= p <= 1:
        raise ValueError("quantile p must lie in [0,1]")
    s = sorted(values)
    i = (len(s) - 1) * p
    low = math.floor(i)
    high = math.ceil(i)
    return s[low] + (s[high] - s[low]) * (i - low)


def summarize(records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    success = [r for r in records if r["status"] == "ok"]
    failures = [r for r in records if r["status"] != "ok"]
    ttft = [r["ttft_ms"] for r in success if r["ttft_ms"] is not None]
    latency = [r["latency_ms"] for r in success]
    tpot = [r["tpot_ms"] for r in success if r["tpot_ms"] is not None]
    usage_complete = bool(success) and all(r["completion_tokens"] is not None for r in success)
    return {
        "sent": len(records),
        "succeeded": len(success),
        "failed": len(failures),
        "wall_time_s": elapsed,
        "requests_per_second": len(success) / elapsed if elapsed > 0 else None,
        "p50_latency_ms": quantile(latency, 0.5),
        "p95_latency_ms": quantile(latency, 0.95),
        "p50_ttft_ms": quantile(ttft, 0.5),
        "p95_ttft_ms": quantile(ttft, 0.95),
        "p50_tpot_ms": quantile(tpot, 0.5) if usage_complete else None,
        "p95_tpot_ms": quantile(tpot, 0.95) if usage_complete else None,
        "output_tokens_per_second": sum(r["completion_tokens"] for r in success) / elapsed if usage_complete and elapsed > 0 else None,
        "usage_coverage": sum(r["completion_tokens"] is not None for r in success) / len(success) if success else 0.,
        "ttft_coverage": len(ttft) / len(success) if success else 0.,
        "error_types": {kind: sum(r.get("error_type") == kind for r in failures) for kind in sorted(set(r.get("error_type", "error") for r in failures))},
    }


def _endpoint(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("base URL must have http(s) scheme and nonempty host")
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _request(index: int, *, endpoint: str, model: str, prompt: str,
             max_tokens: int, timeout: float, seed: int | None) -> dict[str, Any]:
    unique_prompt = f"{prompt}\nTask id: {index:04d}. Answer in one short sentence."
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": unique_prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        payload["seed"] = seed + index
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    # Optional bearer token read only from environment; never include it in logs.
    token = os.environ.get("LLM_OPT_SERVING_API_KEY")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
    start = time.perf_counter()
    first_content: float | None = None
    token_usage: int | None = None
    content_chars = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" not in content_type:
                raise ValueError(f"non-SSE response content type: {content_type}")
            for event in iter_sse_json(response):
                if "error" in event:
                    raise RuntimeError("provider returned SSE error")
                text = chunk_text(event)
                if text:
                    first_content = first_content or time.perf_counter()
                    content_chars += len(text)
                value = completion_tokens(event)
                if value is not None:
                    token_usage = value
        finished = time.perf_counter()
        if first_content is None:
            raise ValueError("no streamed content tokens (TTFT unavailable)")
        generated_after_first = token_usage - 1 if token_usage is not None else 0
        return {
            "index": index, "status": "ok",
            "latency_ms": (finished - start) * 1000.,
            "ttft_ms": (first_content - start) * 1000.,
            "tpot_ms": (finished - first_content) * 1000. / generated_after_first if generated_after_first > 0 else None,
            "completion_tokens": token_usage,
            "completion_chars": content_chars,
        }
    except Exception as exc:
        finished = time.perf_counter()
        return {
            "index": index, "status": "error", "latency_ms": (finished - start) * 1000.,
            "ttft_ms": None, "tpot_ms": None, "completion_tokens": None,
            "error_type": type(exc).__name__,
            # Deliberately omit response bodies, prompts and auth headers.
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True, help="Exactly the loaded model ID from vLLM")
    p.add_argument("--prompt", default="What is a transformer architecture?")
    p.add_argument("--requests", type=int, default=24)
    p.add_argument("--warmup-requests", type=int, default=2,
                   help="Sequential warmup requests excluded from metrics")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/serving_benchmark.json"))
    args = p.parse_args(argv)
    if any(x <= 0 for x in (args.requests,args.concurrency,args.max_tokens,args.timeout)):
        p.error("requests/concurrency/max-tokens/timeout must be positive")
    if args.warmup_requests < 0:
        p.error("warmup-requests must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        endpoint = _endpoint(args.base_url)
    except ValueError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    warmup = [
        _request(-(i + 1), endpoint=endpoint, model=args.model,
                 prompt=args.prompt, max_tokens=args.max_tokens,
                 timeout=args.timeout, seed=args.seed)
        for i in range(args.warmup_requests)
    ]
    if any(r["status"] != "ok" for r in warmup):
        print("BLOCKED: serving endpoint failed warmup requests", file=sys.stderr)
        return 2
    begin = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_request, i, endpoint=endpoint, model=args.model,
                               prompt=args.prompt, max_tokens=args.max_tokens,
                               timeout=args.timeout, seed=args.seed) for i in range(args.requests)]
        records = sorted((f.result() for f in as_completed(futures)), key=lambda r: r["index"])
    elapsed = time.perf_counter() - begin
    metrics = summarize(records, elapsed)
    result = {
        "status": "measured" if metrics["succeeded"] else "blocked",
        "kind": "serving_sse_benchmark",
        "config": {"base_url": args.base_url, "model": args.model,
                   "prompt_chars": len(args.prompt), "requests": args.requests,
                   "warmup_requests": args.warmup_requests,
                   "warmup_not_included_in_metrics": True,
                   "concurrency": args.concurrency, "max_tokens": args.max_tokens,
                   "timeout_seconds": args.timeout, "seed": args.seed},
        "environment": environment(),
        "summary": metrics,
        "requests": records,
    }
    write_json_atomic(args.output, result)
    print(f"Serving: ok={metrics['succeeded']}/{args.requests}, "
          f"p95_latency_ms={metrics['p95_latency_ms']}, "
          f"p95_ttft_ms={metrics['p95_ttft_ms']}, "
          f"token_usage_coverage={metrics['usage_coverage']}")
    if not metrics["succeeded"]:
        print("BLOCKED: endpoint returned no successful responses; see error_types", file=sys.stderr)
        return 2
    return 0 if not metrics["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
