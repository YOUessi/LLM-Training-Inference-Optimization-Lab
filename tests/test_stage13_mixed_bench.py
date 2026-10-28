"""Stage13 identical-request, mixed-output and source-domain baseline contracts."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage13_mixed_bench import (
    DOMAINS,OUTPUT_MODES,POLICIES,RATES,group_modes,make_offers,parse_args,
)
from llm_opt_lab.serving.stage8_arrivals import frozen_requests

ROOT=Path(__file__).resolve().parents[1]


class Stage13MixedBenchContracts(unittest.TestCase):
    def test_both_frozen_source_domains_have_24_verified_requests(self):
        self.assertEqual(set(DOMAINS),{"k8s","prose"})
        for name,(path,manifest,seeds) in DOMAINS.items():
            with self.subTest(domain=name):
                source,m=frozen_requests(ROOT/path,ROOT/manifest)
                self.assertEqual(len(source),24)
                self.assertEqual(len(seeds),2)
                self.assertEqual(len(m["request_jsonl_sha256"]),64)
                for s in seeds:
                    items=make_offers(source,s)
                    self.assertEqual(len(items),24)
                    self.assertEqual(len({x["request_id"] for x in items}),24)
                    self.assertEqual(sum(x["output_mode"]=="label" for x in items),12)
                    self.assertEqual(sum(x["output_mode"]=="detailed" for x in items),12)
                    self.assertEqual(
                        [(x["request_id"],x["output_mode"]) for x in items],
                        [(x["request_id"],x["output_mode"]) for x in make_offers(source,s)],
                    )

    def test_different_seeds_not_same_request_order(self):
        for name,(source,index,seeds) in DOMAINS.items():
            rows,_=frozen_requests(ROOT/source,ROOT/index)
            a=make_offers(rows,seeds[0])
            b=make_offers(rows,seeds[1])
            self.assertNotEqual([x["request_id"] for x in a],
                                [x["request_id"] for x in b])

    def test_closed_predeclared_policy_rate_seed_matrix(self):
        self.assertEqual(POLICIES,("unlimited","fixed150","fixed400","adaptive"))
        self.assertEqual(RATES,(8.,16.))
        for domain in DOMAINS:
            seed=DOMAINS[domain][2][0]
            a=parse_args(["--model","qwen","--domain",domain,
                          "--rate","16","--seed",str(seed),
                          "--policy","adaptive","--output","/tmp/stage13.json"])
            self.assertEqual(a.rate,16.)
            self.assertEqual(a.policy,"adaptive")
            with self.assertRaises(SystemExit):
                parse_args(["--model","qwen","--domain",domain,
                            "--rate","16","--seed","123",
                            "--policy","adaptive","--output","/tmp/x.json"])

    def test_complete_mock_server_case_persists_walltime_for_recomputation(self):
        # This is a purely local MOCK server. It checks JSON provenance
        # and does not stand in for real CUDA/real vLLM performance data.
        import tempfile
        from unittest.mock import patch
        from llm_opt_lab.serving.stage13_mixed_bench import main
        from llm_opt_lab.serving.stage10_admission import accounting
        source,_=frozen_requests(ROOT/DOMAINS["k8s"][0],ROOT/DOMAINS["k8s"][1])
        assigned=make_offers(source,6101)

        def simulated_request(index,**kw):
            mode=("label" if index<0 or index%2==0 else "detailed")
            return {
                "index":index,"status":"ok",
                "prompt_tokens":assigned[index]["target_prompt_tokens"] if index>=0 else 175,
                "completion_tokens":3 if mode=="label" else 128,
                "ttft_ms":8.,"latency_ms":38. if mode=="label" else 200.,
            }

        def fake_monitor(_url,_event,cache,queue):
            cache.append({"kv_usage":0.})
            queue.append({"waiting":0.,"running":0.})

        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"fake_trace.json"
            with patch("llm_opt_lab.serving.stage13_mixed_bench._request",
                       side_effect=simulated_request), patch(
                "llm_opt_lab.serving.stage13_mixed_bench.sample_kv_metrics",
                side_effect=fake_monitor,
            ):
                self.assertEqual(main([
                    "--model","simulated-only","--domain","k8s",
                    "--seed","6101","--rate","16",
                    "--policy","adaptive","--output",str(out),
                ]),0)
            recorded=json.loads(out.read_text())
            self.assertEqual(recorded["status"],"measured")
            self.assertGreater(recorded["elapsed_s"],1.)
            self.assertEqual(
                recorded["summary"],
                accounting(recorded["requests"],recorded["elapsed_s"]),
            )
            self.assertEqual(recorded["summary"]["offered"],24)
            self.assertEqual(recorded["summary"]["rejected"],0)

    def test_mode_breakdown_accepts_rejection_as_slo_failure(self):
        rows=[]
        for i in range(24):
            mode=OUTPUT_MODES[i%2]
            rejected=(i==1 or i==5)
            rows.append({
                "status":"rejected" if rejected else "ok",
                "output_mode":mode,
                "ttft_from_due_ms":None if rejected else 70.,
                "completion_tokens":None if rejected else 3 if mode=="label" else 128,
            })
        result=group_modes(rows)
        self.assertEqual(result["label"]["offered"],12)
        self.assertEqual(result["detailed"]["offered"],12)
        self.assertEqual(result["detailed"]["rejected"],2)
        self.assertEqual(result["detailed"]["completed"],10)
        self.assertEqual(result["label"]["length_gate"]["status"],"pass")
        self.assertEqual(result["detailed"]["length_gate"]["status"],"pass")


if __name__=="__main__":
    unittest.main()
