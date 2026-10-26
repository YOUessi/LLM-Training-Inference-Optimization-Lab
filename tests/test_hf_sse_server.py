"""Offline, CPU-only API validation for the real Transformers SSE adapter."""
from __future__ import annotations

import json
import unittest

from llm_opt_lab.serving.hf_sse_server import (
    MODEL_ID, parse_args, sse_bytes, validate_request,
)
from llm_opt_lab.serving.sse import chunk_text, completion_tokens, iter_sse_json


class ServerContractTests(unittest.TestCase):
    def request(self):
        return {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Short explanation please."}],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    def test_valid_request(self):
        msg, count = validate_request(self.request())
        self.assertEqual(msg[0]["role"], "user")
        self.assertEqual(count, 32)

    def test_reject_unbounded_or_wrong_model(self):
        cases = [
            {"model": "incorrect"},
            {"max_tokens": 129},
            {"max_tokens": 0},
            {"max_tokens": True},
            {"temperature": 0.5},
            {"stream": False},
            {"messages": [{"role": "tool", "content": "x"}]},
            {"messages": [{"role": "user", "content": ""}]},
            {"messages": [{"role": "user", "content": "x"*5000}]},
            {"stream_options": {"include_usage": False}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                data = self.request()
                data.update(overrides)
                with self.assertRaises(ValueError):
                    validate_request(data)

    def test_usage_and_chunks_follow_sse_client_contract(self):
        event = {"choices":[{"index":0,"delta":{"content":"one"}}]}
        usage = {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 7}}
        raw = sse_bytes(event) + sse_bytes(usage) + sse_bytes("[DONE]")
        stream = list(iter_sse_json(raw.splitlines(keepends=True)))
        self.assertEqual(len(stream), 2)
        self.assertEqual(chunk_text(stream[0]), "one")
        self.assertEqual(completion_tokens(stream[1]), 7)

    def test_port_validation(self):
        self.assertEqual(parse_args(["--port", "8026"]).port, 8026)
        with self.assertRaises(SystemExit):
            parse_args(["--port", "80"])


if __name__ == "__main__":
    unittest.main()
