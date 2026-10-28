"""Stage10 analysis fail-closed tests using explicitly synthetic test fixtures."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage10_analysis import SOURCE_SHA, analyze
from llm_opt_lab.serving.stage10_admission import accounting
from llm_opt_lab.serving.stage8_arrivals import output_length_checks

EVIDENCE=Path(__file__).resolve().parents[1]/"docs/evidence/stage9_worker_isolation/real_sse"


def fixtures(root: Path) -> None:
    for seed in (42,137):
        for mode in ("label","detailed"):
            parent=json.loads(
                (EVIDENCE/f"{mode}_w32_rate8_seed{seed}.json").read_text()
            )
            for policy in ("unlimited","bounded"):
                # These are synthetic fixture transformations, not GPU data.
                requests=json.loads(json.dumps(parent["requests"]))
                for r in requests:
                    r["admission_queue_wait_ms"]=0.5
                    if policy=="bounded" and mode=="detailed" and r["index"]>=21:
                        r.update({
                            "status":"rejected",
                            "rejection_reason":"deadline_expired",
                            "ttft_from_due_ms":None,
                            "ttft_ms":None,
                            "latency_ms":None,
                            "prompt_tokens":None,
                            "completion_tokens":None,
                            "admission_queue_wait_ms":250.,
                        })
                completed=[r for r in requests if r["status"]=="ok"]
                cfg={
                    "mode":mode,"policy":policy,"seed":seed,
                    "fixed_offered_rate":8.,
                    "frozen_requests_sha256":
                        "c6d7481c858401c65194f3dfc7d794d510eadf750882be9d8b663489210f237c",
                    "max_client_workers":32,"server_max_num_seqs":8,
                    "bounded_inflight":8,"bounded_queue_slots":4,
                    "admission_deadline_ms":250.,
                    "output_cap_tokens":128,
                    "schedule":"seeded_fixed_duration_exponential_spacings",
                    "schedule_offsets_s":parent["config"]["request_arrival_schedule_s"],
                }
                raw={
                    "status":"measured","kind":"stage10_real_client_admission_policy_study",
                    "environment":{"git_commit":SOURCE_SHA},
                    "config":cfg,"elapsed_s":4.,
                    "accepted_output_length_gate":output_length_checks(mode,completed),
                    "summary":accounting(requests,4.),
                    "server_queue_monitor":{"available":True,"samples":5,
                                            "max_waiting":0,"max_running":8},
                    "gpu_kv_monitor":{"available":True,"samples":5,
                                      "max_usage_fraction":.01},
                    "requests":requests,
                }
                (root/f"{policy}_{mode}_seed{seed}.json").write_text(
                    json.dumps(raw),encoding="utf-8"
                )


class Stage10AnalysisContracts(unittest.TestCase):
    def test_matched_pairs_include_rejection_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixtures(root)
            a=analyze(root)
            self.assertEqual(a["status"],"pass")
            self.assertEqual(a["total_offered_requests"],192)
            self.assertEqual(len(a["paired_cases"]),4)
            detailed=[p for p in a["paired_cases"] if p["mode"]=="detailed"]
            self.assertEqual(len(detailed),2)
            for p in detailed:
                self.assertEqual(p["bounded"]["rejected"],3)
                self.assertEqual(p["unlimited"]["rejected"],0)
                self.assertIsNone(p["bounded"]["all_offered_p95_ttft_ms"])

    def test_mismatched_arrival_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixtures(root)
            p=root/"bounded_detailed_seed42.json"
            d=json.loads(p.read_text())
            d["config"]["schedule_offsets_s"][0]=.17
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(root)

    def test_unaccounted_request_status_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixtures(root)
            p=root/"bounded_detailed_seed137.json"
            d=json.loads(p.read_text())
            d["summary"]["offered"]=23
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(root)

    def test_missing_real_case_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixtures(root)
            (root/"unlimited_label_seed42.json").unlink()
            with self.assertRaises(FileNotFoundError):
                analyze(root)


if __name__=="__main__":
    unittest.main()
