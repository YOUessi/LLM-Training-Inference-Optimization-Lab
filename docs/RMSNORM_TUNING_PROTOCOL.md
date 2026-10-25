# RMSNorm microbenchmark tuning protocol (2026-10-26)

## Question

Can a grouped-row Triton RMSNorm kernel improve the negative results
recorded in [RESULTS_2026-10-26.md](RESULTS_2026-10-26.md),
especially `128x4096 FP16` and `256x4096 FP32`?

The old one-row kernel lost against **properly fused** `torch.compile`.
One favorable run does not demonstrate a general win.

## Candidate design

- One Triton CTA/program handles 1, 2, 4 or 8 independent rows.
- Test `num_warps=1,2,4,8` with `BLOCK=next_power_of_2(columns)`.
- Enforce max 32768 tensor elements in one CTA to reduce memory pressure.
- Each candidate must pass `torch.testing.assert_close` at the original dtype.
- Floating-point square accumulation remains FP32. No learned affine weight or residual.
- Compare with:
  1. eager PyTorch composed baseline (historical only);
  2. *compiled* PyTorch with dtype conversion **inside** the compiled function;
  3. original one-row Triton kernel.

## Time measurement

`triton.testing.do_bench_cudagraph` minimizes host dispatch overhead
for microsecond kernels. Report repeated **independent timed trial outputs**,
not a single best kernel time. Randomize candidate/baseline interleaving.
Preserve all candidate latencies and failed configurations.

## Primary evaluation

```bash
PYTHONPATH=src python -m llm_opt_lab.kernels.rmsnorm_tune \
  --rows 128 --cols 4096 --dtype float16 --trials 4 --rep-ms 35 \
  --output results/rmsnorm_tune_128x4096_f16.json
PYTHONPATH=src python -m llm_opt_lab.kernels.rmsnorm_tune \
  --rows 256 --cols 4096 --dtype float32 --trials 4 --rep-ms 35 \
  --output results/rmsnorm_tune_256x4096_f32.json
```

Pick one robust configuration, then run it at a **later time** with
different random seed / invocation order and additional shapes (not used
in tuning). Only then consider dispatch policy changes.

## Interpreting output

`best_in_sample` is a *candidate*, not a verified regression fix.
Measurements are specific to GPU, PyTorch, Triton, tensor shape and dtype.
If custom Triton remains slower than Inductor, retain the failure and
identify the performance bottleneck instead of changing the baseline.

The output includes `environment.git_commit`, numerical error and raw
repetitions. Never edit JSON to fabricate speedups.
