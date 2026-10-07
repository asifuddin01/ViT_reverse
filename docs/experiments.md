# Experiment log

## CIFAR-10 small-model baseline

**Status:** the full seed-7, 20-epoch CPU run completed on 2026-10-07. The best validation accuracy was **69.50% at epoch 20** (validation cross-entropy **0.8529**). Training accuracy at that epoch was **72.65%**. The 20 recorded epochs took **1,509.9 seconds (25.2 minutes)** in total, excluding setup and final checkpoint verification. A fresh model loaded from the best checkpoint reproduced the saved validation logits exactly and yielded the same 69.50% accuracy. This is one seed on one held-out validation split, not a test-set result or an uncertainty estimate.

The training task uses the official [torchvision CIFAR-10](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CIFAR10.html) training split (50,000 images). A deterministic stratified 90/10 split with seed 7 creates 45,000 training and 5,000 validation examples; the official test split stays untouched. The exact selected indices are stored in the run directory. Training uses random crop with four-pixel padding, random horizontal flip, tensor conversion, and fixed channel normalization `(x-0.5)/0.5`. Validation uses only tensor conversion and normalization.

The CPU-feasible scratch model in `configs/vit_tiny_cifar.yaml` has 32×32 input, 4×4 patches, embedding width 96, four blocks, eight heads, and ten classes. The protocol is AdamW, learning rate `3e-4`, weight decay `0.05`, batch size 128, two warm-up epochs followed by cosine decay, and 20 epochs. Best model selection uses validation accuracy only. The run saves `config.yaml`, `split_indices.json`, `environment.json`, per-epoch `metrics.csv`, resumable `last.pt`, best-validation `checkpoint.pt`, and `summary.json`. The trainer checks that validation logits match exactly after reloading the best checkpoint.

```bash
vit-lab train --config configs/vit_tiny_cifar.yaml \
  --run-dir results/runs/cifar_tiny_seed7 --device cpu
```

The versioned measurements are [`cifar_tiny_seed7_metrics.csv`](../results/tables/cifar_tiny_seed7_metrics.csv), [`cifar_tiny_seed7_summary.json`](../results/tables/cifar_tiny_seed7_summary.json), and the [loss/accuracy curve](../results/figures/cifar_tiny_seed7_training.svg). Regenerate the curve with:

```bash
python scripts/plot_training.py results/runs/cifar_tiny_seed7/metrics.csv \
  results/figures/cifar_tiny_seed7_training.svg
```

`results/runs/` and `data/` are ignored by Git. The launch environment manifest recorded commit `d415e20` because the trainer changes were still uncommitted at launch. Those same source files were committed as `2311889` while the run was active, without modifying them during training. The local run directory holds its exact split indices, config, full checkpoints, and environment manifest. The official CIFAR-10 test split remains untouched. This training model serves implementation and later ablation studies; it is separate from the 224-pixel pretrained ViT-Base reference-equivalence model.

## Preregistered patch-size experiment

The first CPU-feasible controlled ablation varies native CIFAR-10 patch size **4, 8, 16** at fixed image size 32, width 96, depth 4, eight heads, and ten classes. Seeds **7, 11, 19** are paired across patch sizes: a given seed sets both model randomness and the same deterministic 45,000/5,000 split for every patch variant. All variants retain the baseline augmentation, batch size 128, AdamW settings, 20-epoch warmup/cosine schedule, and best-validation checkpoint rule. The completed `P=4, seed=7` baseline is reused. This narrower width-96 matrix is an initial CPU study; the width-192 matrix in the execution plan remains pending. No official test images will be used to choose a variant.

The nine frozen configurations live in `configs/experiments/patch_size/` plus the existing baseline config. Verify them with `python scripts/run_patch_ablation.py --prepare-only`. The serial runner `python scripts/run_patch_ablation.py --run` skips complete runs, resumes interrupted runs from `last.pt`, and writes validated small evidence to `results/ablations/patch_size/`. Model checkpoints and full split lists remain in ignored `results/runs/`.

All nine 20-epoch runs completed. The official CIFAR-10 test split was never used. Each best checkpoint reproduced its saved validation predictions exactly after reload. For each seed, the three variants have the same saved train/validation split hash; different seeds produce different splits.

| Patch size | Seed 7 | Seed 11 | Seed 19 | Mean ± sample SD | Sum of epoch time |
|---|---:|---:|---:|---:|---:|
| 4×4 | 69.50% | 72.84% | 72.42% | **71.59 ± 1.82%** | 71.6 min |
| 8×8 | 62.96% | 64.70% | 62.54% | **63.40 ± 1.15%** | 30.7 min |
| 16×16 | 54.98% | 56.46% | 56.16% | **55.87 ± 0.78%** | 21.8 min |

The [aggregate CSV](../results/ablations/patch_size/aggregate.csv) gives unrounded values; the [accuracy figure](../results/figures/patch_size_ablation.svg) displays the mean and sample standard deviation. Per-run configs, split hashes, environments, best epochs, training durations, and 20-row metrics are in [the ablation results directory](../results/ablations/patch_size/). The three-seed means are descriptive, not confidence intervals. Because seed controls both initialization and the 45,000/5,000 split, the SD includes both sources of variation. In this protocol and these three seeds, patch size 4 had higher validation accuracy than 8, and 8 higher than 16. This does not establish a universal patch-size rule or a test-set ranking.

The seed-7 best checkpoints were also compared on a separate CPU FP32, batch-1 inference run with 10 warmups, 50 timed forwards, four PyTorch threads, and a fresh process per architecture. A fixed random 32×32 input excludes preprocessing. The sampled process RSS includes the Python/PyTorch runtime and allocator, so it is not activation-only memory.

| Patch size | Tokens incl. CLS | Parameters | Analytical MACs/image | Median latency | Sampled process RSS |
|---|---:|---:|---:|---:|---:|
| 4×4 | 65 | 459,562 | 32.29 M | 1.120 ms | 314 MB |
| 8×8 | 17 | 468,778 | 8.04 M | 0.856 ms | 295 MB |
| 16×16 | 5 | 522,922 | 2.53 M | 0.719 ms | 281 MB |

The [cost CSV](../results/ablations/patch_size/cost.csv) and [JSON with all timing trials](../results/ablations/patch_size/cost.json) contain exact values and hardware metadata. MACs use the convention in `docs/benchmarking.md`; latency and RSS are local CPU measurements. Larger patches reduce token-dependent computation here, while patch-projection parameter count grows. The latency difference is much smaller than the MAC difference because fixed runtime overhead matters at this model size. Regenerate the figures and cost table with:

```bash
python scripts/plot_patch_ablation.py results/ablations/patch_size/aggregate.csv \
  results/figures/patch_size_ablation.svg
python scripts/benchmark_patch_ablation.py --warmups 10 --trials 50 --threads 4
```

The plot script requires the `experiments` extra from `pyproject.toml` (or `matplotlib` installed separately). The width-192 matrix, other ablations, and official test-set evaluation remain future work.

## Preregistered head-count experiment

This CPU-feasible study fixes native 32×32 CIFAR-10, 8×8 patches, width 96, depth 4, all training settings above, and paired seeds 7/11/19. It varies only the number of attention heads: **4, 8, 12, 16**, all divisors of 96. The three completed `P=8, heads=8` patch-size runs are reused without retraining. The nine new configs are frozen in `configs/experiments/head_count/` and can be verified with `python scripts/run_head_ablation.py --prepare-only`. The serial runner `python scripts/run_head_ablation.py --run` resumes incomplete runs, checks exact seed-paired split hashes and best-checkpoint reloads, and exports small evidence to `results/ablations/head_count/`.

All twelve 20-epoch cases completed. Every best checkpoint reproduced its validation predictions after reload, and all four variants within a seed have the same saved train/validation split hash. The official CIFAR-10 test split was not used.

| Heads | Seed 7 | Seed 11 | Seed 19 | Mean ± sample SD | Sum of epoch time |
|---|---:|---:|---:|---:|---:|
| 4 | 63.08% | 64.70% | 63.50% | **63.76 ± 0.84%** | 25.3 min |
| 8 | 62.96% | 64.70% | 62.54% | **63.40 ± 1.15%** | 30.7 min |
| 12 | 62.72% | 63.62% | 62.54% | **62.96 ± 0.58%** | 35.0 min |
| 16 | 63.10% | 63.54% | 61.96% | **62.87 ± 0.82%** | 41.9 min |

The [aggregate CSV](../results/ablations/head_count/aggregate.csv) gives unrounded values; the [accuracy figure](../results/figures/head_count_ablation.svg) displays mean and sample SD. Each run's config hash, split hash, environment, best epoch, training duration, and 20-row metrics are in [the head-count results directory](../results/ablations/head_count/). The three-seed means and SDs are descriptive, not confidence intervals. Seed changes both initialization and the validation split. Four heads had the highest observed mean by 0.36 percentage points over eight heads, while the seed-level differences were small and varied. This is a result for this width-96, patch-8, 20-epoch protocol, not a general rule about ViT head count. The width-192 study in the execution plan remains pending.

The seed-7 best checkpoints were measured separately for CPU FP32 batch-1 inference: 10 warmups, 50 timed forwards, four PyTorch threads, and a fresh process per head count. A random 32×32 input excludes preprocessing. Parameter and analytical MAC counts are identical because width, depth, patch size, and token count are fixed. Theoretical storage for all four layers' FP32 attention maps increases with head count; sampled process RSS includes Python/PyTorch runtime and allocator behavior and is not an activation-only measure.

| Heads | Parameters | Analytical MACs/image | Median latency | Sampled process RSS | All-layer attention maps |
|---|---:|---:|---:|---:|---:|
| 4 | 468,778 | 8.04 M | 0.648 ms | 287 MB | 18.5 kB |
| 8 | 468,778 | 8.04 M | 0.732 ms | 308 MB | 37.0 kB |
| 12 | 468,778 | 8.04 M | 0.771 ms | 333 MB | 55.5 kB |
| 16 | 468,778 | 8.04 M | 0.823 ms | 302 MB | 74.0 kB |

The [cost CSV](../results/ablations/head_count/cost.csv) and [JSON with all trials](../results/ablations/head_count/cost.json) contain exact values and hardware metadata. Latency rose with head count in this local trial despite identical analytical MACs; runtime scheduling and measurement noise may contribute. Process RSS is nonmonotonic and should not be read as exact attention memory. Reproduce the figures and cost table with:

```bash
python scripts/plot_head_ablation.py results/ablations/head_count/aggregate.csv \
  results/figures/head_count_ablation.svg
python scripts/plot_head_ablation.py results/ablations/head_count/aggregate.csv \
  results/figures/head_count_ablation.png
python scripts/benchmark_head_ablation.py --warmups 10 --trials 50 --threads 4
```

## Preregistered position-embedding experiment

The next paired study asks whether learned absolute positions improve the same native 32×32 CIFAR-10 model with 8×8 patches, width 96, depth 4, eight heads, and CLS pooling. The two modes are **learned** and **none**. In `none`, neither the CLS token nor any of the 16 patch tokens receives a position vector; the CLS token itself remains. The model has no `pos_embed` parameter in that mode. Learned mode has a `[1,17,96]` position parameter, so its parameter count is 468,778 versus 467,146 without positions. The position-free initializer consumes an unused draw of the same shape so every shared initial weight stays identical when the same seed is used; a model test verifies this. The position-free model's CLS prediction is also invariant to swapping complete image patches in a small test case.

The three completed `P=8, heads=8` runs for seeds 7/11/19 are the learned-position baseline and are reused without retraining. The three new position-free configs are frozen in `configs/experiments/position/`. Both variants use the same 45,000/5,000 deterministic split per seed, training augmentation, AdamW hyperparameters, batch size 128, two warm-up epochs, 20-epoch cosine schedule, and best-validation checkpoint rule. The official CIFAR-10 test set remains untouched. `python scripts/run_position_ablation.py --prepare-only` verifies frozen configs; `python scripts/run_position_ablation.py --run` resumes incomplete runs and exports split hashes, 20-row histories, checkpoint-reload evidence, and mean/sample SD to `results/ablations/position/`. Inspect `progress.json` there after each run. This section is a protocol, with no new position-free accuracy claimed yet.
