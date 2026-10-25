"""Streaming OpenAI-compatible SSE event parsing (network-free testable logic)."""
from __future__ import annotations

import json
from typing import Any, Iterable, Iterator


def iter_sse_json(lines: Iterable[bytes | str]) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from SSE frames separated by a blank line.

    SSE allows multiple data lines per event. Ignore keepalive comments and
    non-data fields (event/id/retry). Invalid JSON raises ValueError.
    """
    data: list[str] = []

    def emit() -> dict[str, Any] | None:
        payload = "\n".join(data).strip()
        data.clear()
        if not payload or payload == "[DONE]":
            return None
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("expected an SSE JSON object")
        return value

    for original in lines:
        line = original.decode("utf-8") if isinstance(original, bytes) else original
        line = line.rstrip("\r\n")
        if not line:
            value = emit()
            if value is not None:
                yield value
            continue
        if line.startswith(":"):
            continue
        if line == "data" or line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    value = emit()
    if value is not None:
        yield value


def chunk_text(event: dict[str, Any]) -> str:
    """Extract visible streamed text; non-text tool/function events do not count as TTFT."""
    choices = event.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta") or {}
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict) and isinstance(x.get("text"), str))
    return ""


def completion_tokens(event: dict[str, Any]) -> int | None:
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None
    n = usage.get("completion_tokens")
    # Ensure a real provider token count, not an approximate SSE chunk count.
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 0 else None
