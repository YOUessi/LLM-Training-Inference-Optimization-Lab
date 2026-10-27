"""Validate pinned NVTX/CUDA profile and distinguish traces from perf claims."""
from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "stage3_nsys_nvtx"


class Stage3NvtxTests(unittest.TestCase):
    def setUp(self):
        self.manifest=json.loads((ROOT/"manifest.json").read_text(encoding="utf-8"))

    def test_source_and_correctness(self):
        m=self.manifest
        d=json.loads((ROOT/"workload.json").read_text(encoding="utf-8"))
        self.assertEqual(d["status"],"measured")
        self.assertEqual(d["kind"],"real_cuda_rmsnorm_nvtx_trace_driver")
        self.assertTrue(d["not_an_uninstrumented_speed_benchmark"])
        self.assertEqual(d["environment"]["git_commit"],m["source_commit"])
        self.assertEqual(d["config"]["repeats"],30)
        self.assertEqual(set(d["cases"]),{
            "eager_unfused","torch_inductor_fused","custom_triton",
        })
        for c in d["cases"].values():
            self.assertEqual(c["correctness"],"pass")
            self.assertEqual(c["iterations"],30)
            self.assertGreaterEqual(c["max_absolute_error"],0)

    def test_gpu_ops_and_kernel_count(self):
        m=self.manifest
        self.assertEqual({
            k:v["gpu_ops"] for k,v in m["nvtx_ranges"].items()
        },{"eager_unfused":150,"torch_inductor_fused":30,"custom_triton":30})
        for k in ("custom_triton","torch_inductor_fused"):
            self.assertEqual(m["primary_kernel"][k]["instances"],30)
            self.assertGreater(m["primary_kernel"][k]["mean_us"],0)

    def test_nvidia_trace_stays_local_and_no_hardware_counters(self):
        m=self.manifest
        self.assertFalse(m["hardware_counters_recorded"])
        self.assertFalse(m["report_binary_committed_to_github"])
        self.assertEqual(len(m["report_binary_sha256"]),64)
        self.assertEqual(m["source_commit"],"1ae20a245e5d745f21cb5b976a6fecc76839a92c")

    def test_all_csv_hashes_and_nvtx_names(self):
        m=self.manifest
        for file,sha in m["csv_sha256"].items():
            with self.subTest(file=file):
                self.assertEqual(
                    hashlib.sha256((ROOT/file).read_bytes()).hexdigest(),sha
                )
        with (ROOT/"nvtx_nvtx_gpu_proj_sum.csv").open() as f:
            labels={r["Range"].split(":")[-1]:int(r["Total GPU Ops"])
                    for r in csv.DictReader(f)}
        self.assertEqual(labels,{k:v["gpu_ops"] for k,v in m["nvtx_ranges"].items()})


if __name__=="__main__":
    unittest.main()
