# Stage24 verified Qwen3-8B-AWQ vLLM engine capacity experiment

Source: real Tang RTX 4090 Laptop single-GPU vLLM 0.19.1. Qwen3-8B-AWQ, thinking and prefix caching OFF, 0.82 GPU memory utilization and 2048 model context. The only treatment is `max-num-seqs` 8 versus 16, with unlimited client admission. Two frozen CPython-stdlib arrival seeds (9301,9309) at 8 and 16 offered requests/s, 24 offers per arm, counterbalanced cap order = 8 cells / **192 real offers**. Exactly identical input IDs and scheduled arrival offsets across paired capacity settings, zero request errors and zero explicit rejection. Original eight JSONs, SHA256 and source SHA are archived in `evidence/stage24`.

## Aggregated across two arrival seeds, 48 requests per server-capacity/rate

| Offered rate | Server max-num-seqs | All-offered TTFT <=250ms | Useful output tokens/sec (mean) |
|---|---:|---:|---:|
| 8/s | 8 | 47.9% | 248.16 |
| 8/s | 16 | **52.1%** | **271.22** |
| 16/s | 8 | 12.5% | 263.99 |
| 16/s | 16 | **14.6%** | **314.49** |

At 16/s, doubling server concurrency raised average useful output throughput about 19.1% (314.49/263.99) with only a modest +2.1 percentage-point improvement in 250ms SLO. Under seed9301/16s the sampled server waiting peak fell from 12 to 6; seed9309/16s from 9 to 4. This demonstrates actual server-capacity pressure, not a guarantee of optimal batch sizing. At 8/s, throughput improved about 9.3%. One high-rate seed's SLO250 was unchanged, hence no claim of uniform latency benefit.

**Decision:** server-capacity 16 can improve delivered throughput but **neither configuration meets a strong all-offered 250ms SLO under 16/s**. No production rollout or statistical significance claim. Limitations: very small 24-offer cases, only two arrival seeds, sequential sessions, synthetic technical questions on PSF-licensed stdlib source, one quantized 8B model and one 16GiB GPU. More realistic SLO curves and new capacity/quality exploration are needed.

Independent computation: `analysis/stage24_capacity.py`. Paired raw evidence: `evidence/stage24`.
