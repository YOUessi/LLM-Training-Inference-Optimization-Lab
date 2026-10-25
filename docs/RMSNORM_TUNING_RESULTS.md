# RMSNorm grouped-row configuration sweep — 2026-10-26

## Scope and reproducibility

- Real device: RTX 4090 Laptop (SM 8.9), Torch 2.10.0+cu128, Triton 3.6.0.
- Candidate source commit: `db191d5`. This is an **experimental configuration sweep**, not a deployed kernel policy.
- Original grouped RMSNorm experiments: 16 configurations per shape (4 row groups × 4 warp settings); all candidates passed PyTorch correctness.
- Repeated timings use `triton.testing.do_bench_cudagraph`, not the earlier non-CUDA-graph `do_bench`; do not combine their latency numbers.
- `torch.compile` includes the final output dtype conversion in the traced function.
- Validation: fixed chosen warp count/row count, **new random seed (3571)**, five trials and two additional shapes not included in tuning.

## In-sample configuration search

| Shape | Tested | Best rows/program | Best warps | Best GPU latency (ms) | Speedup vs compiled |
|---|---:|---:|---:|---:|---:|
| 128×4096 float16 | 16 | 1 | 8 | 0.001562 | 1.078× |
| 256×4096 float32 | 16 | 1 | 8 | 0.003770 | 0.949× |
| 128×1024 float16 | 16 | 1 | 4 | 0.001108 | 1.014× |

## Fixed-candidate independent validation

| Shape | Rows/program | Warps | Median GPU latency (ms) | Speedup vs compiled | Outcome |
|---|---:|---:|---:|---:|---|
| 128×4096 float16 | 1 | 8 | 0.001561 | 1.078× | Faster than compiled |
| 256×4096 float32 | 1 | 8 | 0.003878 | 0.924× | Slower than compiled |
| 128×1024 float16 | 1 | 4 | 0.001264 | 1.017× | Near parity |
| 64×2048 float16 | 1 | 4 | 0.001338 | 1.120× | Faster than compiled |
| 512×8192 float16 | 1 | 8 | 0.007202 | 0.956× | Near parity |

## Decision

**Do not enable grouped-row RMSNorm dispatch by default.** Contrary to the hypothesis, the fastest in-sample candidates all used **one row per program**; wider FP32/large FP16 remain at or below compiled PyTorch speed in independent validation.

The 1.08× FP16 midwidth result is small and hardware/timing-sensitive. It is insufficient evidence to claim reliable improvement in LLM serving.

## Raw evidence

- [`rmsnorm_tune_128x4096_f16.json`](evidence/rmsnorm_tune_128x4096_f16.json)
- [`rmsnorm_tune_256x4096_f32.json`](evidence/rmsnorm_tune_256x4096_f32.json)
- [`rmsnorm_tune_128x1024_f16.json`](evidence/rmsnorm_tune_128x1024_f16.json)
- [`rmsnorm_validate_128x4096_float16_w8.json`](evidence/rmsnorm_validate_128x4096_float16_w8.json)
- [`rmsnorm_validate_256x4096_float32_w8.json`](evidence/rmsnorm_validate_256x4096_float32_w8.json)
- [`rmsnorm_validate_128x1024_float16_w4.json`](evidence/rmsnorm_validate_128x1024_float16_w4.json)
- [`rmsnorm_validate_64x2048_float16_w4.json`](evidence/rmsnorm_validate_64x2048_float16_w4.json)
- [`rmsnorm_validate_512x8192_float16_w8.json`](evidence/rmsnorm_validate_512x8192_float16_w8.json)

## Issues and follow-up

- Candidate selection is subject to winner's curse and few runs; use a larger held-out shape suite if dispatch changes are proposed.
- The small absolute microsecond times are CUDA-graph device timings, not Python launch latency or full model tokens/sec.
- Without Nsight Compute occupancy/DRAM counters, the mechanism of loss vs Inductor is not fully established.
- A higher-value next milestone is actual vLLM service request latency and throughput, then operator profiling.
