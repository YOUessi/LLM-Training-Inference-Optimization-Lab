# Stage25–29 single-batch engineering delivery and remaining gates

**Completed 2026-10-29 on Tang single RTX4090 Laptop GPU.** No claim of asynchronous work by ChatGPT, remote work happened through verified Tang/connected GitHub tools during the conversation. Original old 10-minute Stage13 cron loop repeatedly failed with `SOURCE_DRIFT`: removed safely after backing up its crontab, no unrelated cron lines affected. New tasks run as finite **pinned-detached-worktree** jobs with exact expected Git SHA and a shared file lock `/home/you/.cache/llm_opt_shared_gpu.lock`, GPU process and port preflight, fixed and guarded task choices, non-overwriting results, failure propagation and evidence retention. The finite jobs executed sequentially on the one physical GPU while CPU/data/analysis work was developed separately; no dangerous overlapping GPU jobs were started.

## Evidence matrix

| Track | Real execution | Acceptance | Main evidence |
|---|---|---|---|
| Stage25 vLLM | 22 GPU cells ×24 = 528 accepted and completed 8B-AWQ requests. PSF licensed 24 newly sourced Python documentation prompts in 4 context buckets, shared and distinct prefixes; 8/16 offers/s; cache on/off; max-seqs 8/16; GPU utilization .7/.82; output cap128/256 | PASS **experimental conditional performance only**; shared-prefix cache ON yielded +92.3% useful TPS and +63pp SLO250, distinct-prefix control did not; not a production workload | [Result](STAGE25_MEASURED_RESULTS.md) [raw](../evidence/stage25_measured) |
| Stage26 real SFT | Upstream SHA-verified Dolly 15k CC BY-SA human responses; 512 train / 96 val / 96 test; 5 8B NF4 QLoRA configs (rank, BF16/FP16, accum, GC), full frozen test selection | PASS real training and test: B rank16 selected on val; heldout assistant CE2.815→1.637, PPL16.69→5.14; BF16 GC allocated peak memory12.72→10.03GiB with ~27% throughput cost | [Result](STAGE26_MEASURED_RESULTS.md) [raw](../evidence/stage26_sft_v2) |
| Stage27 CUDA | Eight real Qwen3-shaped fused residual+weighted RMSNorm and materialized masked attention Softmax shape/dtype comparisons vs optimized `torch.compile`; PyTorch CUDA profiler on 2 | PASS microbenchmark/correctness; conditional 1.14× FP16/1.18× BF16 Inductor relative speedup at RMSNorm 256×4096; no evidence of end-to-end LLM improvement | [Result](STAGE27_MEASURED_RESULTS.md) [raw](../evidence/stage27_measured) |
| Stage28 training infra | Four separate 12-step real Transformer LoRA training runs: no checkpoint, full-state sync, full-state async, step6 restart from verified async; 10 original snapshots validated | PASS exact adapter SHA, optimizer SHA and final loss after restart; **async is not faster** than no checkpoint in this small task | [Result](STAGE28_MEASURED_RESULTS.md) [raw](../evidence/stage28_measured) |
| Stage29 quality | 96 deterministic greedy Qwen3-8B base vs fixed rank16 adapter test pairs on frozen Dolly test | PASS lexical-reference measurement; ROUGE-L F1 .240→.367, word F1 .321→.437, creative-writing worse | [Result](STAGE29_MEASURED_RESULTS.md) [raw](../evidence/stage29_measured) |

## Preserved negative runs

- Stage25 v1 13 prior cases failed output-length gate; the old raw JSON/sha are under `evidence/stage25_invalid_v1`, never merged into Stage25 v2 measured summary.
- Stage26 microbatch2/no-GC 8B QLoRA OOM is preserved under `evidence/stage26_invalid_v1`; its configs excluded and safe auxiliary microbatch1 comparison identified with distinct optimizer budget.
- Stage27 initial Triton `do_bench` return type bug and original stack trace are preserved under `evidence/stage27_invalid_v1`.
- Stage29 one shell parser failure and one missing attention-mask warning/interrupted attempt are preserved under `evidence/stage29_invalid_v1/v2`; final 96-case run fixed and frozen.

## Explicitly NOT completed

- Production-like cold/warm prefix distribution and multiple independent seeds with uncertainty intervals (synthetic reused-header shared-prefix workload gives unusually large gains).
- Human/independent-model semantic judgment, multi-seed SFT quality and training over all 512 available source records; tested SFT uses 192 model exposures and single checkpoint selection.
- Integrated end-to-end Qwen3 kernel replacement or Nsight Compute hardware occupancy/DRAM counters; standalone causal softmax is not FlashAttention.
- Actual multi-GPU DDP/FSDP FULL_SHARD, distributed checkpoint/restart: hardware currently one 16GiB GPU, **blocked**, cannot claim verified without another machine.

**Go/no-go:** research artifact evidence passes all automated contracts; production deployment and published generalization claims remain blocked. Run `PYTHONPATH=src:. python3 -m unittest discover -s tests -q` plus `PYTHONPATH=src:. python3 analysis/stage25_28_batch_acceptance.py --root . --out results/stage25_29_audit.json` and verify GitHub remote SHA. Raw artifacts and source are the factual record.
