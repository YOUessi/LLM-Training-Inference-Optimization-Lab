# Stage33: Qwen3-8B-AWQ real 59.375-second sustained offered-window prefix-cache study

**Measured** 12 real vLLM cases × 96 unique offered=**1,152 completed real requests**, exactly 59.375-second scheduled arrival window at 1.6/s, three seeded schedules and two engineered prefix mixtures (cold0 all unique leading headers, hot50 48/96 intentionally shared long header); prefix cache ON/OFF, fresh model server process for each arm. All-offered SLO250/500/1000/2000, TTFT P95/P99, useful output TPS, real server waiting/running gauges and KV fraction retained in raw JSON `evidence/stage33_sustained/`. Every paired case preserves request identity/text SHA/arrival offsets. No unreported rejection/errors.

| Prefix mixture | Caching ON vs OFF useful TPS | Seed-bootstrap exploratory 95% interval | Mean P95 arrival TTFT change | Mean all-offered SLO500 change | TPS improved in |
|---|---:|---:|---:|---:|---:|
| cold0 | +0.00% | -0.00% to +0.01% | +1.7ms | +0.35pp | 2/3 |
| hot50 | +0.08% | +0.01% to +0.12% | -72.4ms | +5.56pp | 3/3 |

Compared with Stage30's 8/s and 11.875-second offered window, Stage33 is a lower offered rate (1.6/s) for ~60 seconds. These regimes confound duration and load and **cannot** by themselves attribute any difference solely to elapsed time. At lower load a 50% engineered hot prefix still helps useful output throughput (~8%) and TTFT, whereas fully cold prefix does not. Actual vLLM prefix-cache hit ratio was not measured, and three arrival seeds do not justify population confidence. The bootstrap resamples seed-level paired means, not thousands of correlated requests as independent treatments. These tests are not a multi-hour production soak.

