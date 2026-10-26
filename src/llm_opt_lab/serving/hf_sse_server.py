"""Loopback-only, real Transformers model serving over OpenAI-compatible SSE.

This server is a diagnostic baseline. It uses one shared model and a serialized
generation lock: NO vLLM, dynamic batching, paged attention, or request scheduler.
It must never be reported as a vLLM benchmark.
"""
from __future__ import annotations

import argparse
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from llm_opt_lab.serving.hf_baseline import find_cached_model

MODEL_ID = "qwen2.5-1.5b-instruct-transformers-local"
MAX_BODY_BYTES = 65536
MAX_PROMPT_CHARS = 4096
MAX_NEW_TOKENS = 128


def validate_request(data: Any, model_id: str = MODEL_ID) -> tuple[list[dict[str, str]], int]:
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    if data.get("model") != model_id:
        raise ValueError("model ID mismatch")
    if data.get("stream") is not True:
        raise ValueError("stream=True required")
    if data.get("stream_options", {}).get("include_usage") is not True:
        raise ValueError("stream_options.include_usage=True required")
    n = data.get("max_tokens", 32)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_NEW_TOKENS:
        raise ValueError("max_tokens must be 1..128")
    if data.get("temperature", 0.0) not in (0, 0.0):
        raise ValueError("only deterministic temperature=0 is supported")
    messages = data.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 8:
        raise ValueError("messages must contain 1..8 entries")
    clean: list[dict[str, str]] = []
    total_chars = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("unsupported message role")
        text = message.get("content")
        if not isinstance(text, str) or not text:
            raise ValueError("only nonempty text messages supported")
        total_chars += len(text)
        clean.append({"role": message["role"], "content": text})
    if total_chars > MAX_PROMPT_CHARS:
        raise ValueError("prompt too long")
    return clean, n


def sse_bytes(payload: dict[str, Any] | str) -> bytes:
    value = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) if isinstance(payload, dict) else payload
    return ("data: " + value + "\n\n").encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "LLMOptBaseline/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        # No prompt text, URLs with query params, authorization or completions logged.
        return

    def send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(404, {"error": "not found"})
            return
        self.send_json(200, {
            "status": "ready",
            "model": MODEL_ID,
            "engine": "transformers-not-vllm",
            "dynamic_batching": False,
        })

    def _emit(self, payload: dict[str, Any] | str) -> None:
        self.wfile.write(sse_bytes(payload))
        self.wfile.flush()

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        try:
            value = self.headers.get("Content-Length")
            length = int(value) if value is not None else 0
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("request body length invalid")
            data = json.loads(self.rfile.read(length))
            messages, max_tokens = validate_request(data)
        except (ValueError, json.JSONDecodeError, TypeError):
            self.send_json(400, {"error": "invalid bounded stream request"})
            return

        runtime = self.server.runtime
        # A single model, no continuous batching: serialize, so queueing is
        # included in client-observed TTFT as it should be.
        with runtime["lock"]:
            try:
                tokenizer = runtime["tokenizer"]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
                input_len = int(encoded["input_ids"].shape[1])
                if input_len > 1024:
                    self.send_json(413, {"error": "prompt token limit exceeded"})
                    return
                import torch
                from transformers import TextIteratorStreamer
                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True,
                    timeout=45,
                )
                result: dict[str, Any] = {}

                def generate() -> None:
                    try:
                        with torch.inference_mode():
                            result["sequences"] = runtime["model"].generate(
                                **encoded,
                                streamer=streamer,
                                max_new_tokens=max_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                use_cache=True,
                            )
                    except Exception:
                        result["error"] = "model generation failed"
                        # Release the streamer consumer if generation died.
                        streamer.on_finalized_text("", stream_end=True)

                worker = threading.Thread(target=generate, daemon=True)
                worker.start()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                sent_content = False
                try:
                    for fragment in streamer:
                        if fragment:
                            sent_content = True
                            self._emit({
                                "object": "chat.completion.chunk",
                                "choices": [{"index": 0, "delta": {"content": fragment},
                                             "finish_reason": None}],
                            })
                    worker.join(timeout=20)
                    if worker.is_alive() or "error" in result or "sequences" not in result:
                        self._emit({"error": "model generation incomplete"})
                        self._emit("[DONE]")
                        return
                    token_count = int(result["sequences"].shape[-1]) - input_len
                    self._emit({
                        "object": "chat.completion.chunk",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": input_len,
                            "completion_tokens": token_count,
                            "total_tokens": input_len + token_count,
                        },
                    })
                    self._emit("[DONE]")
                except (BrokenPipeError, ConnectionResetError, queue.Empty):
                    # Client cancellation/slow generation. Do not claim success.
                    return
            except Exception:
                # Do not leak raw prompts, model paths, stack traces or secrets.
                # If headers were already sent, close the connection.
                if not self.wfile.closed:
                    try:
                        self.send_json(500, {"error": "inference unavailable"})
                    except (BrokenPipeError, OSError):
                        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8026)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.75)
    a = p.parse_args(argv)
    if not 1024 <= a.port <= 65535:
        p.error("port must be in 1024..65535")
    if not 0.1 <= a.gpu_memory_fraction <= 0.95:
        p.error("gpu-memory-fraction must be between .1 and .95")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA GPU is required")
        path = a.model_dir or find_cached_model()
        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(path), dtype=torch.float16, local_files_only=True,
            trust_remote_code=False,
        ).to("cuda").eval()
        server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
        server.daemon_threads = True
        server.runtime = {
            "model": model, "tokenizer": tokenizer, "lock": threading.Lock(),
        }
        print(
            f"SERVER_READY engine=transformers-not-vllm model={MODEL_ID} "
            f"endpoint=127.0.0.1:{a.port}", flush=True
        )
        try:
            server.serve_forever(poll_interval=0.1)
        finally:
            server.server_close()
        return 0
    except Exception as exc:
        print(f"BLOCKED: model streaming server unavailable: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
