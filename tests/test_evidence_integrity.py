"""Validate that committed experiment evidence is structured and non-fabricated.

These checks validate consistency and provenance only; they do not rerun CUDA.
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parent.parent / "docs" / "evidence"


def read_strict(path: Path):
    def reject_constant(value: str):
        raise ValueError(f"nonfinite number in evidence: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


class EvidenceIntegrityTests(unittest.TestCase):
    def test_cuda_kernel_benchmarks_have_checks_and_raw_times(self):
        files = sorted(EVIDENCE.glob("kernel_*.json"))
        self.assertGreaterEqual(len(files), 8)
        for path in files:
            with self.subTest(file=path.name):
                report = read_strict(path)
                self.assertEqual(report["status"], "measured")
                self.assertEqual(report["kind"], "gpu_kernel_benchmark")
                self.assertTrue(report["environment"]["cuda_available"])
                self.assertEqual(len(report["environment"]["git_commit"]), 40)
                for case in report["cases"].values():
                    self.assertEqual(case["correctness"], "pass")
                    self.assertGreaterEqual(case["max_absolute_error"], 0)
                    self.assertTrue(math.isfinite(case["median_speedup"]))
                    for impl in ("torch", "triton"):
                        self.assertGreater(case[impl]["median_ms"], 0)
                        self.assertGreaterEqual(case[impl]["p95_ms"], case[impl]["median_ms"])
                    if report.get("compiled_baseline_requested", False):
                        self.assertGreater(case["compiled_torch"]["median_ms"], 0)
                        self.assertTrue(math.isfinite(case["median_speedup_vs_compiled"]))

    def test_candidate_swaps_and_validation_use_real_gpu(self):
        tune = sorted(EVIDENCE.glob("rmsnorm_tune_*.json"))
        fixed = sorted(EVIDENCE.glob("rmsnorm_validate_*.json"))
        self.assertEqual(len(tune), 3)
        self.assertEqual(len(fixed), 5)
        for path in tune + fixed:
            with self.subTest(file=path.name):
                d = read_strict(path)
                self.assertEqual(d["status"], "measured")
                self.assertTrue(d["environment"]["cuda_available"])
                self.assertTrue(d["experimental_not_deployed"])
                self.assertEqual(d["parameters"]["timing_method"],
                                 "triton.testing.do_bench_cudagraph")
                self.assertGreaterEqual(d["parameters"]["trials"], 4)
                self.assertTrue(d["candidates"])
                self.assertEqual(d["best_in_sample"]["median_ms"],
                                 min(r["median_ms"] for r in d["candidates"]))
                self.assertTrue(all(r["correctness"] == "pass" for r in d["candidates"]))
                self.assertEqual(len(d["candidates"]), 16 if path in tune else 1)

    def test_offline_inference_evidence_is_not_mislabeled_as_vllm(self):
        report = read_strict(EVIDENCE / "hf_offline_qwen2_5_1_5b.json")
        self.assertEqual(report["status"], "measured")
        self.assertEqual(report["kind"],
                         "offline_pytorch_transformers_generation_not_vllm")
        self.assertEqual([r["batch_size"] for r in report["results"]], [1, 2, 4])
        self.assertTrue(report["environment"]["cuda_available"])
        for row in report["results"]:
            self.assertGreater(row["p50_latency_ms"], 0)
            self.assertGreater(row["total_output_tokens_per_second"], 0)
            self.assertGreater(row["max_peak_allocated_gib"], 0)
            self.assertIsNone(row["ttft_ms"])
            self.assertIsNone(row["tpot_ms"])

    def test_profiler_does_not_claim_nsight_hardware_counters(self):
        p = read_strict(EVIDENCE / "rmsnorm_profiler_256x4096_f32.json")
        self.assertTrue(p["not_nsight_hardware_counters"])
        self.assertEqual(p["status"], "measured")
        self.assertEqual(set(p["profile"]),
                         {"torch_inductor", "triton_original", "triton_row1_warps"})


if __name__ == "__main__":
    unittest.main()
