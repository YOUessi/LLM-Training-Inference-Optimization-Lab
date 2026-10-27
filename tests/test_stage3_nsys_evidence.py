"""Validate genuine Nsight Systems CUDA evidence and permission boundaries.

This suite validates pinned evidence only. A CPU test must never pretend
to have collected NVIDIA performance counters or reproduced GPU traces.
"""
from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "stage3_nsys"


class NsightSystemsEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))

    def test_pinned_version_and_source_commit(self):
        m = self.manifest
        self.assertEqual(m["schema"], "nsight-systems-kernel-trace-v1")
        self.assertEqual(
            m["source_commit"],
            "2749782737d3873e103314814a4f62cf1cd19248",
        )
        self.assertEqual(m["nsight_systems_cli_version"],
                         "2026.5.1.161-265138896106v0")
        self.assertEqual(
            m["nsys_cli_package_sha256"],
            "61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08",
        )

    def test_real_cuda_launches_have_measured_kernel_data(self):
        m = self.manifest
        p = json.loads((EVIDENCE / "kernel.json").read_text(encoding="utf-8"))
        self.assertEqual(p["status"], "measured")
        self.assertEqual(p["kind"], "gpu_kernel_benchmark")
        self.assertTrue(p["environment"]["cuda_available"])
        self.assertEqual(p["environment"]["git_commit"], m["source_commit"])
        self.assertEqual(p["tensor"]["dtype"], "float16")
        self.assertEqual((p["tensor"]["rows"], p["tensor"]["cols"]), (256, 4096))
        for result in p["cases"].values():
            self.assertEqual(result["correctness"], "pass")

    def test_kernel_stats_and_checksum(self):
        path = EVIDENCE / "kernel_stats_cuda_gpu_kern_sum.csv"
        data = path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(),
                         self.manifest["csv_sha256"])
        rows = list(csv.DictReader(data.decode("utf-8").splitlines()))
        self.assertGreater(len(rows), 5)
        for label in ("_rmsnorm_kernel", "_softmax_kernel", "cunn_SoftMaxForwardReg"):
            matching = [r for r in rows if label in r["Name"]]
            self.assertTrue(matching, label)
            self.assertGreater(int(matching[0]["Instances"]), 0)
            self.assertGreater(float(matching[0]["Avg (ns)"]), 0)

    def test_hardware_counter_limitations_explicit(self):
        m = self.manifest
        self.assertFalse(m["hardware_sm_or_dram_metrics_collected"])
        self.assertEqual(m["cpu_perf_event_paranoid"], 4)
        self.assertEqual(m["nvidia_RmProfilingAdminOnly"], 1)
        self.assertFalse(m["raw_binary_trace_committed_to_github"])
        self.assertEqual(len(m["raw_nsys_rep_sha256"]), 64)
        self.assertEqual(len(m["csv_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
