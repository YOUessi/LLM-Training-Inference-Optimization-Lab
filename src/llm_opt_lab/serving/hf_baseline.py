"""Real offline Hugging Face/PyTorch generation baseline (NOT vLLM).

Use cached Qwen2.5-1.5B-Instruct on a NVIDIA GPU and report prefill+decode
wall time, request throughput, output-token throughput and peak GPU memory.
TTFT/TPOT for streaming are intentionally not claimed by this path.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import sys
import time
from pathlib import Path

from llm_opt_lab.provenance import environment, write_json_atomic


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--batch-sizes", default="1,2,4")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/hf_offline_baseline.json"))
    a = p.parse_args(argv)
    try:
        bs = tuple(int(x.strip()) for x in a.batch_sizes.split(","))
    except ValueError:
        p.error("--batch-sizes requires comma-separated positive integers")
    if not bs or len(set(bs)) != len(bs) or any(x < 1 or x > 16 for x in bs):
        p.error("--batch-sizes must contain distinct integers from 1 to 16")
    a.batch_sizes = bs
    if a.max_new_tokens <= 0 or a.max_new_tokens > 256 or a.warmup < 1 or a.repeats < 2:
        p.error("max-new-tokens=1..256, warmup>=1, repeats>=2")
    return a


def quantile(xs: list[float], p: float) -> float:
    ys = sorted(xs)
    i = (len(ys)-1)*p
    lo, hi = math.floor(i), math.ceil(i)
    return ys[lo] + (ys[hi]-ys[lo])*(i-lo)


def find_cached_model() -> Path:
    roots = sorted(
        (Path.home()/".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots").glob("*")
    )
    valid = [p for p in roots if p.is_dir() and (p/"config.json").is_file() and
             (p/"model.safetensors").is_file() and (p/"tokenizer.json").is_file()]
    if not valid:
        raise FileNotFoundError("offline model Qwen2.5-1.5B-Instruct is not cached")
    if len(valid) > 1:
        raise ValueError("multiple cached model revisions; pass explicit --model-dir")
    return valid[0]


def actual_generated_tokens(generated, eos_ids: set[int]) -> int:
    """Count emitted tokens until the first EOS inclusive, excluding padding after EOS."""
    tokens = generated.tolist()
    for i, tok in enumerate(tokens):
        if tok in eos_ids:
            return i + 1
    return len(tokens)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        print(f"BLOCKED: torch and transformers required: {exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("BLOCKED: must run on real NVIDIA CUDA GPU", file=sys.stderr)
        return 2
    try:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        path = a.model_dir or find_cached_model()
        if not (path/"config.json").is_file():
            raise FileNotFoundError(f"model config missing: {path}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float16,
            local_files_only=True, trust_remote_code=False
        ).to("cuda").eval()
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        eos = model.generation_config.eos_token_id
        eos_ids = {int(x) for x in eos} if isinstance(eos, (list,tuple)) else ({int(eos)} if eos is not None else set())
        prompt_base = (
            "Explain how a transformer attention layer works, provide a short "
            "example, and mention one limitation. Give a compact explanation."
        )
        results = []
        for batch_size in a.batch_sizes:
            dialogs = [
                [{"role":"user", "content":prompt_base + f" Task {i+1}."}]
                for i in range(batch_size)
            ]
            texts = [tokenizer.apply_chat_template(d,tokenize=False,add_generation_prompt=True) for d in dialogs]
            encoded = tokenizer(texts, padding=True,return_tensors="pt").to("cuda")
            input_width = int(encoded["input_ids"].shape[1])
            kwargs = {"max_new_tokens":a.max_new_tokens,
                      "do_sample":False, "use_cache":True,
                      "pad_token_id":tokenizer.pad_token_id}
            with torch.inference_mode():
                for _ in range(a.warmup):
                    model.generate(**encoded,**kwargs)
            torch.cuda.synchronize()
            latency_ms = []
            generated_totals = []
            peak_bytes = []
            with torch.inference_mode():
                for _ in range(a.repeats):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    sequences = model.generate(**encoded,**kwargs)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    output_ids = sequences[:,input_width:].detach().cpu()
                    tokens = sum(actual_generated_tokens(seq,eos_ids) for seq in output_ids)
                    latency_ms.append(elapsed*1000.)
                    generated_totals.append(int(tokens))
                    peak_bytes.append(int(torch.cuda.max_memory_allocated()))
            if not all(n>0 for n in generated_totals):
                raise RuntimeError("one or more trials generated zero output tokens")
            elapsed_sum_s = sum(latency_ms)/1000.
            results.append({
                "batch_size":batch_size,
                "input_tokens_per_request_padded":input_width,
                "latency_ms_per_trial":latency_ms,
                "actual_output_tokens_per_trial":generated_totals,
                "gpu_peak_allocated_bytes_per_trial":peak_bytes,
                "p50_latency_ms":statistics.median(latency_ms),
                "p95_latency_ms":quantile(latency_ms, 0.95),
                "total_output_tokens_per_second":sum(generated_totals)/elapsed_sum_s,
                "requests_per_second":batch_size*a.repeats/elapsed_sum_s,
                "max_peak_allocated_gib":max(peak_bytes)/(1024**3),
                "ttft_ms":None,
                "tpot_ms":None,
            })
            print(f"bs={batch_size} p50={results[-1]['p50_latency_ms']:.1f}ms "
                  f"output_tps={results[-1]['total_output_tokens_per_second']:.1f} "
                  f"peak_GiB={results[-1]['max_peak_allocated_gib']:.2f}")
        payload = {
            "status":"measured",
            "kind":"offline_pytorch_transformers_generation_not_vllm",
            "model":{"id":"Qwen2.5-1.5B-Instruct",
                     "hf_snapshot":path.name,
                     "weights":"local_files_only"},
            "experiment":{"batch_sizes":a.batch_sizes,
                          "max_new_tokens":a.max_new_tokens,
                          "repeats":a.repeats,"warmup":a.warmup,
                          "seed":a.seed,
                          "prompt_sha256":hashlib.sha256(prompt_base.encode()).hexdigest(),
                          "timing_excludes_tokenization":True,
                          "timing_includes_prefill_and_decode":True},
            "environment":environment(),
            "results":results,
        }
        write_json_atomic(a.output,payload)
        print("Report:",a.output)
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
