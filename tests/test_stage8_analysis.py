"""Pure CPU tests for strict 288-request Stage8 result acceptance.

Temporary mocked result files are test fixtures, NOT claimed GPU data.
"""
from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from llm_opt_lab.serving.stage8_analysis import SOURCE_SHA, CORPUS_SHA, analyze
from llm_opt_lab.serving.stage8_arrivals import MODES, RATES, SEEDS

INPUTS=Path(__file__).resolve().parents[1]/"data/stage8_frozen_requests.jsonl"
ROWS=[json.loads(x) for x in INPUTS.read_text(encoding="utf-8").splitlines()]


def fixtures(root: Path) -> None:
    for seed in SEEDS:
        source=list(ROWS)
        random.Random(seed).shuffle(source)
        for mode in MODES:
            for rate in RATES:
                responses=[]
                for i,item in enumerate(source):
                    n={"label":3,"summary":60,"detailed":128}[mode]
                    responses.append({
                        "index":i, "status":"ok",
                        "request_id":item["request_id"],
                        "bucket":item["bucket"],"language":item["language"],
                        "topic":item["topic"],"prompt_sha256":item["prompt_sha256"],
                        "prompt_tokens":item["target_prompt_tokens"],
                        "target_prompt_tokens":item["target_prompt_tokens"],
                        "completion_tokens":n,"ttft_ms":40.,
                        "ttft_from_due_ms":60.,
                        "client_executor_wait_ms":20.,
                    })
                summary={
                    "succeeded":24,"failed":0,
                    "prompt_usage_coverage":1., "usage_coverage":1.,
                    "p95_ttft_ms":40., "p95_latency_ms":100.,
                    "output_tokens_per_second":100. if rate==2 else 200.,
                }
                case={
                    "status":"measured",
                    "kind":"real_vllm_stage8_multilingual_openloop",
                    "environment":{"git_commit":SOURCE_SHA},
                    "config":{
                        "seed":seed,"mode":mode,"rate_per_second":rate,
                        "frozen_request_sha256":CORPUS_SHA,
                        "output_cap_tokens":128,"requests":24,
                    },
                    "summary":summary,
                    "all_actual_input_tokens_in_bounds":True,
                    "length_gates":{"status":"pass","realized_mean":
                                    {"label":3.,"summary":60.,"detailed":128.}[mode],
                                    "fraction_stopped_before_cap":0 if mode=="detailed" else 1},
                    "requests":responses,"p95_ttft_from_scheduled_arrival_ms":60.,
                    "p95_executor_queue_wait_ms":20.,
                    "completed_requests_per_second":2.,
                    "offered_arrival_rate_empirical_per_second":rate,
                    "actual_input_token_min":170,
                    "actual_input_token_max":680,
                    "ttft_slo_violations":{str(i):0 for i in (100,250,500,1000)},
                    "gpu_kv_cache_gauge":{"peak_fraction":.1,"samples":15},
                }
                (root/f"{mode}_rate{int(rate)}_seed{seed}.json").write_text(
                    json.dumps(case),encoding="utf-8",
                )


class Stage8AnalysisContracts(unittest.TestCase):
    def test_12_matrix_cases_all_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixtures(p)
            r=analyze(p)
            self.assertEqual(r["measured_requests"],288)
            self.assertEqual(r["case_count"],12)
            self.assertEqual(len(r["descriptive_two_seed_cells"]),6)
            self.assertEqual(len(r["experimental_slo_admission_decisions"]),12)
            self.assertTrue(r["not_a_new_gpu_experiment"])

    def test_bad_matched_source_trace_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixtures(p)
            target=p/"detailed_rate8_seed42.json"
            d=json.loads(target.read_text())
            d["requests"][0]["prompt_sha256"]="bad"
            target.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(p)

    def test_failed_usage_is_not_reported_as_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixtures(p)
            target=p/"summary_rate2_seed137.json"
            d=json.loads(target.read_text())
            d["summary"]["prompt_usage_coverage"]=0.5
            target.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                analyze(p)

    def test_missing_case_cannot_be_estimated(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);fixtures(p)
            (p/"label_rate2_seed137.json").unlink()
            with self.assertRaises(FileNotFoundError):
                analyze(p)


if __name__=="__main__":
    unittest.main()
