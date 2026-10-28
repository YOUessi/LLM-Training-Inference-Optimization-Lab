"""Stage13 pre-measurement CPU fail-closed analyst checks.

The temporary fixture files in these tests are MOCK data for software
contracts only, NOT GPU evidence or publishable performance claims.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage13_analysis import analyze
from llm_opt_lab.serving.stage13_mixed_bench import (
    DOMAINS,RATES,POLICIES,make_offers,group_modes,
)
from llm_opt_lab.serving.stage8_arrivals import frozen_requests,schedule
from llm_opt_lab.serving.stage10_admission import accounting

SRC="a"*40
ROOT=Path(__file__).resolve().parents[1]


def fixture(root:Path):
    for domain,(corpus,manifest,seeds) in DOMAINS.items():
        original,meta=frozen_requests(ROOT/corpus,ROOT/manifest)
        for seed in seeds:
            offers=make_offers(original,seed)
            for rate in RATES:
                offsets=schedule(seed,rate,24,kind="fixed_window")
                for policy in POLICIES:
                    records=[]
                    for i,item in enumerate(offers):
                        mode=item["output_mode"]
                        n=3 if mode=="label" else 128
                        records.append({
                            "index":i,"request_id":item["request_id"],
                            "language":item["language"],"topic":item["topic"],
                            "bucket":item["bucket"],"output_mode":mode,
                            "prompt_sha256":item["prompt_sha256"],
                            "target_prompt_tokens":item["target_prompt_tokens"],
                            "prompt_tokens":item["target_prompt_tokens"],
                            "completion_tokens":n,"status":"ok",
                            "ttft_from_due_ms":80.,
                            "ttft_ms":80.,"latency_ms":300.,
                            "scheduled_offset_s":offsets[i],
                            "client_executor_wait_ms":.01,
                            "admission_queue_wait_ms":.05,
                            "policy_decision":{
                                "policy_name":policy,
                                "predicted_service_ms":50. if mode=="label" else 1000.,
                                "dynamic_admission_deadline_ms":0. if policy=="unlimited" else 150.,
                                "active_at_arrival":0,"waiting_at_arrival":0,
                            }
                        })
                    elapsed=4. if rate==8 else 2.
                    payload={
                        "status":"measured",
                        "kind":"stage13_mixed_real_vllm_adaptive_admission",
                        "environment":{"git_commit":SRC},
                        "config":{
                            "domain":domain,"seed":seed,"rate_per_second":rate,
                            "policy":policy,
                            "frozen_requests_sha256":meta["request_jsonl_sha256"],
                            "max_client_workers":32,"server_max_num_seqs":8,
                            "mode_count":{"label":12,"detailed":12},
                            "arrival_schedule_offsets_s":offsets,
                        },
                        "summary":accounting(records,elapsed),
                        "elapsed_s":elapsed,
                        "actual_mode_summaries":group_modes(records),
                        "gate_final_state":{"active":0,"waiters":0},
                        "server_queue_monitor":{"available":True,"samples":4,
                                                "max_waiting":0},
                        "kv_cache_monitor":{"available":True,"samples":4},
                        "requests":records,
                    }
                    folder=root/f"stage13_{domain}_seed{seed}"
                    folder.mkdir(parents=True,exist_ok=True)
                    (folder/f"rate{int(rate)}_{policy}_seed{seed}.json").write_text(
                        json.dumps(payload),encoding="utf-8"
                    )


class Stage13AnalysisContracts(unittest.TestCase):
    def test_full_768_software_fixture_and_honest_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            d=analyze(root,SRC)
            self.assertEqual(d["replayed_offered_requests"],768)
            self.assertEqual(d["real_case_count"],32)
            self.assertEqual(len(d["paired_scenarios"]),8)
            self.assertEqual(len(d["two_seed_summary"]),4)
            self.assertFalse(any(x["candidate_win_vs_unlimited"]
                                 for x in d["paired_scenarios"]))
            self.assertTrue(d["not_a_new_gpu_run"])

    def test_mismatched_source_or_missing_one_policy_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            with self.assertRaises(ValueError):
                analyze(root,"wrong")
            p=root/"stage13_prose_seed7201/rate16_adaptive_seed7201.json"
            p.unlink()
            with self.assertRaises(FileNotFoundError):
                analyze(root,SRC)

    def test_changed_prompt_for_one_policy_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            p=root/"stage13_k8s_seed6101/rate16_adaptive_seed6101.json"
            d=json.loads(p.read_text())
            d["requests"][0]["prompt_sha256"]="f"*64
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(root,SRC)

    def test_rejected_offer_must_change_all_offered_slo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            p=root/"stage13_prose_seed7209/rate8_fixed150_seed7209.json"
            d=json.loads(p.read_text())
            r=d["requests"][5]
            r["status"]="rejected"
            r["rejection_reason"]="deadline_expired"
            r["ttft_from_due_ms"]=None
            r["prompt_tokens"]=None
            r["completion_tokens"]=None
            # Deliberately *do not* recompute aggregate. Must fail.
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(root,SRC)


if __name__=="__main__":
    unittest.main()
