# Stage34: truly previously unused Dolly128 IDs — frozen three-model reference CE test

The original **Dolly 15k** dataset SHA-256 and license attribution are frozen in `data/stage34_manifest.json`. 128 category-stratified human-written instruction/answer examples, 16 per each of eight categories, selected deterministically by SHA-256 *after excluding all 704 previously selected Stage26/31 train+validation+test identities*. No test-based model/adapter selection, no additional training; evaluate the same Qwen3-8B BF16 NF4 base, Stage26 short SFT, and Stage31 full two-epoch best selected on Stage31 validation. Each candidate computes **the same 128 reference answer token-only cross-entropies**, full assistant prompt mask; actual 10,307 answer tokens; exact per-example identity, category, CE and sample count recorded in `evidence/stage34_unused128/fresh128_three_model_CE.json`.

| Model | Previously unused examples | Target answer tokens | CE nats/token | Perplexity |
|---|---:|---:|---:|---:|
| base | 128 | 10307 | 2.63453 | 13.9368 |
| stage26_short | 128 | 10307 | 1.58260 | 4.8676 |
| stage31_full | 128 | 10307 | 1.56132 | 4.7651 |

Full512 SFT vs short192-exposure SFT has a *modest* CE reduction of **-0.02128 nats/token**, with a stratified within-128-sample bootstrap 95% interval [-0.02589, -0.01646]. This interval excludes zero *conditional on the fixed source sample, trained model and one seed*. It does **not** cover training-seed variability or evaluate semantic correctness. The full512 model is also better than untuned base on this reference-likelihood metric. These 128 records are unused by earlier training/validation/test **but from the same publicly available Dolly dataset**, possibly in original Qwen pretraining. This is *record-disjoint*, not independently sourced generalization. No human evaluation or safety-alignment gain claimed.

