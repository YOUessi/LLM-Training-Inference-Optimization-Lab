"""End-to-end transport contract with an in-process fake SSE server.

This is a NETWORK-LOCAL MOCK. Passing this test does not prove real vLLM
availability, TTFT, or GPU inference performance.
"""
from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_opt_lab.serving.bench_client import _request, _endpoint


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length))
        if req.get("stream") is not True or req.get("stream_options", {}).get("include_usage") is not True:
            self.send_error(400)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in [
            {"choices": [{"delta": {"content": "hello"}}]},
            {"choices": [{"delta": {"content": " there"}}]},
            {"choices": [], "usage": {"completion_tokens": 2}},
        ]:
            data = "data: " + json.dumps(event) + "\n\n"
            self.wfile.write(data.encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *_args):
        return


class ServingTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2.0)

    def test_stream_receives_provider_token_count(self):
        endpoint = _endpoint("http://127.0.0.1:" + str(self.server.server_port))
        result = _request(0, endpoint=endpoint, model="mock-only",
                          prompt="hello", max_tokens=8, timeout=5., seed=42)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["completion_chars"], len("hello there"))
        self.assertIsNotNone(result["ttft_ms"])
        self.assertIsNotNone(result["tpot_ms"])


if __name__ == "__main__":
    unittest.main()
