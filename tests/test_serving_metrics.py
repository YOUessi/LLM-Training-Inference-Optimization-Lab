from __future__ import annotations

import json
import unittest

from llm_opt_lab.serving.bench_client import _endpoint, quantile, summarize
from llm_opt_lab.serving.sse import chunk_text, completion_tokens, iter_sse_json, prompt_tokens


class SSETests(unittest.TestCase):
    def test_stream_parses_json_and_done(self):
        lines = [
            b": keepalive\n", b"event: message\n",
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n', b"\n",
            b'data: {"choices":[],"usage":{"completion_tokens":4}}\n', b"\n",
            b"data: [DONE]\n", b"\n",
        ]
        events = list(iter_sse_json(lines))
        self.assertEqual(len(events), 2)
        self.assertEqual(chunk_text(events[0]), "hi")
        self.assertEqual(completion_tokens(events[1]), 4)

    def test_multi_data_lines(self):
        lines = ['data: {"choices": [],\n', 'data: "usage": {"completion_tokens": 2}}\n','\n']
        self.assertEqual(completion_tokens(next(iter_sse_json(lines))), 2)

    def test_real_prompt_token_usage(self):
        d={"usage":{"prompt_tokens":1024,"completion_tokens":21}}
        self.assertEqual(prompt_tokens(d),1024)
        self.assertEqual(completion_tokens(d),21)
        self.assertIsNone(prompt_tokens({"usage":{"prompt_tokens":True}}))
        self.assertIsNone(prompt_tokens({"usage":None}))

    def test_no_chunk_estimation(self):
        self.assertIsNone(completion_tokens({"choices": [{"delta": {"content": "word"}}]}))
        self.assertEqual(chunk_text({"choices":[{"delta":{"content":"word"}}]}), "word")
        self.assertEqual(chunk_text({"choices":[{"delta":{"content":None}}]}), "")
        self.assertIsNone(completion_tokens({"usage":{"completion_tokens":True}}))

    def test_invalid_json_fails(self):
        with self.assertRaises(json.JSONDecodeError):
            list(iter_sse_json(["data: {broken}\n","\n"]))


class MetricsTests(unittest.TestCase):
    def test_quantile(self):
        self.assertEqual(quantile([10., 20., 30.], 0.5), 20.)
        self.assertEqual(quantile([], 0.95), None)
        self.assertAlmostEqual(quantile([10., 20.], 0.95), 19.5)

    def test_usage_complete(self):
        rs = [
            {"status":"ok", "latency_ms":300., "ttft_ms":100., "tpot_ms":50.,
             "completion_tokens":5},
            {"status":"ok", "latency_ms":200., "ttft_ms":80., "tpot_ms":40.,
             "completion_tokens":4},
        ]
        s = summarize(rs, 1.0)
        self.assertEqual(s["output_tokens_per_second"], 9.0)
        self.assertEqual(s["p50_ttft_ms"], 90.)

    def test_partial_usage_not_faked(self):
        rs = [{"status":"ok", "latency_ms":300., "ttft_ms":100., "tpot_ms":None,
               "completion_tokens":None}]
        s = summarize(rs, 1.)
        self.assertIsNone(s["output_tokens_per_second"])
        self.assertIsNone(s["p50_tpot_ms"])
        self.assertEqual(s["usage_coverage"], 0.)

    def test_failed_requests(self):
        rs = [{"status":"error", "error_type":"URLError", "latency_ms":99.,
               "ttft_ms":None,"tpot_ms":None,"completion_tokens":None}]
        s = summarize(rs, 1.)
        self.assertEqual(s["failed"], 1)
        self.assertEqual(s["error_types"], {"URLError":1})

    def test_endpoint(self):
        self.assertEqual(_endpoint("http://localhost:8000/v1"), "http://localhost:8000/v1/chat/completions")
        self.assertEqual(_endpoint("http://localhost:8000"), "http://localhost:8000/v1/chat/completions")
        with self.assertRaises(ValueError):
            _endpoint("file:///etc/passwd")


if __name__ == "__main__":
    unittest.main()
