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

The nine frozen configurations live in `configs/experiments/patch_size/` plus the existing baseline config. Regenerate or verify them with `python scripts/run_patch_ablation.py --prepare-only`. Resume the serial matrix with `python scripts/run_patch_ablation.py --run`; it skips completed runs, resumes an interrupted run from `last.pt`, and writes small validated per-run evidence plus `results/ablations/patch_size/progress.json`. Model checkpoints and full split lists remain in ignored `results/runs/`. The final mean and sample standard deviation across three seeds will be reported only after every run completes. Training time and the fact that the validation split changes across seed IDs will be reported alongside accuracy.

The first paired seed is complete. Best validation accuracy was **69.50%** for `P=4` (epoch 20), **62.96%** for `P=8` (epoch 19), and **54.98%** for `P=16` (epoch 20). Their exact split hashes match and every best checkpoint reproduced its saved validation predictions after reload. Epoch time summed to 25.2, 10.2, and 7.3 minutes respectively. These are interim single-seed observations; seeds 11 and 19 are still pending, so no uncertainty estimate or general patch-size conclusion is available.

For seed 11, `P=4` reached **72.84%** best validation accuracy at epoch 20 and `P=8` reached **64.70%** at epoch 19. Their saved split hashes match and both checkpoint reloads passed. `P=16` for seed 11 remains pending, so the second paired comparison is not yet complete.
