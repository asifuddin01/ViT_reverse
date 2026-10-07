# Project handoff status

**Updated:** 2026-10-07  
**Project directory:** `/Users/mdasifuddin/AI/ViT_reverse`  
**Branch:** `main`  
**Remote:** `https://github.com/asifuddin01/ViT_reverse.git`  
**Current task:** Run the controlled CIFAR-10 patch-size ablation. The nine configurations and serial resume/export runner are frozen; `P=4, seed=7` is the completed baseline, and the next pending training run is `P=8, seed=7`.

## Completed, in order

1. Repository and Python 3.12 package setup. The original brief is `PROJECT_SPEC.md`; the work plan is `VIT_EXECUTION_PLAN.md`. Dependencies are captured in `requirements-lock.txt`.
2. Pinned timm ViT-Base/16 reference inspected and its `model.safetensors` SHA-256 verified. The checkpoint has 152 tensors and 86,567,656 parameters. See `docs/reference_contract.md` and `vit-lab inspect-reference`.
3. From-scratch patch projection and full inspectable ViT implemented. Patch convolution equals explicit patch extraction in values and input gradients. The network-free timm oracle test passes.
4. The reference's 152 parameters map directly to our model with identical names and shapes. A CPU/FP32 comparison with reference fused attention disabled passed **100 of 100** captured stage comparisons (25 stages × seeded random, zeros, generated RGB, and an official CIFAR-10 image), with recorded maximum absolute difference `0.0` for those inputs. See `docs/reverse_engineering.md` and `results/tables/`.
5. The user-provided GitHub repository is configured as `origin`; `main` was pushed and tracks `origin/main`.
6. Official CIFAR-10 data downloaded and passed torchvision's integrity check. The deterministic stratified train/validation loader, AdamW trainer, checkpoint/resume logic, and CLI are implemented. Two four-batch smoke runs passed, including exact validation-prediction reproduction after checkpoint reload.
7. `pytest -q` passed **18 tests** after the trainer changes.
8. The full 20-epoch seed-7 CIFAR-10 CPU baseline completed. Best validation accuracy was **69.50% at epoch 20**; the official test split was untouched. The best checkpoint reproduced its saved validation logits exactly after reload. Versioned results are `results/tables/cifar_tiny_seed7_metrics.csv`, `results/tables/cifar_tiny_seed7_summary.json`, and `results/figures/cifar_tiny_seed7_training.svg`. See `docs/experiments.md`.
9. Attention and representation diagnostics completed on four fixed official CIFAR-10 training images using the pinned ViT-Base weights. Six selected CLS-to-patch overlays, normalized row checks, bitwise repeatability, and 20 representation rows are saved in `results/analysis/cifar10_seed7/`; methodology and limits are in `docs/attention_analysis.md`. `pytest -q -p no:cacheprovider` passed **20 tests**.
10. Parameter/MAC/FLOP and six-case CPU inference benchmarks completed on Apple M1. The pinned educational and timm models each have 86,567,656 trainable parameters and 346,270,624 FP32 parameter bytes. The educational model was measured at 224/384/512 pixels; timm's fused model was measured at 224. Raw timing samples, process RSS, and formulas are in `results/tables/benchmark.csv` and `.json`; the plot is `results/figures/benchmark_resolution.svg`. See `docs/benchmarking.md` for measurement limits. The analytical count matched PyTorch's profiled matrix/convolution FLOPs in a small-model test. `pytest -q -p no:cacheprovider` passed **22 tests** and `pip check` found no broken requirements.
11. The patch-size ablation's nine configs and resumable serial runner were prepared before new training. The frozen protocol is documented in `docs/experiments.md`. `pytest -q -p no:cacheprovider` passed **23 tests**, including config invariance and freeze checks. No new variant accuracy is claimed yet.

## Current limits and honest interpretation

- The real-image comparison uses a 32×32 CIFAR-10 photograph resized for ViT-Base. It verifies numerical behavior, not native high-resolution detail.
- `torch.backends.mps.is_available()` returned `False` on this host. CPU is the verified device; CUDA and MPS results are unmeasured.
- The CIFAR result has one seed and one validation split. There is no CIFAR test accuracy yet, no uncertainty estimate across training seeds, and no medical transfer or ablation result. Do not claim their results.
- The attention figures use 32×32 photographs enlarged to 224×224 and are diagnostics, not explanations or localization maps. The representation summary covers only four selected images.
- Benchmark timings varied materially under host contention, especially the 512-pixel and educational batch-8 cases. CPU process RSS is sampled and includes runtime/allocator effects; it is not exact device peak allocation. See `docs/benchmarking.md`.
- Model checkpoints and datasets are intentionally excluded from Git. The pinned pretrained checkpoint is in the Hugging Face cache and can be re-downloaded from the configured revision.

## Resume commands

```bash
cd /Users/mdasifuddin/AI/ViT_reverse
source .venv/bin/activate
pytest -q
vit-lab inspect-reference --config configs/vit_base.yaml --offline
vit-lab compare --config configs/vit_base.yaml --offline --device cpu
cat results/runs/cifar_tiny_seed7/summary.json
python scripts/plot_training.py results/runs/cifar_tiny_seed7/metrics.csv results/figures/cifar_tiny_seed7_training.svg
vit-lab analyze --config configs/vit_base.yaml --device cpu --offline
vit-lab benchmark --config configs/vit_base.yaml --offline --threads 4 --warmups 2 --trials 5
python scripts/run_patch_ablation.py --prepare-only
python scripts/run_patch_ablation.py --run --max-runs 1
git status --short --branch
git remote -v
```

If `.venv` is missing, recreate it with Python 3.12 and `python -m pip install -e '.[dev]'`. The first `inspect-reference` without `--offline` downloads the pinned checkpoint. The comparison creates `results/tables/equivalence.csv`, `equivalence.json`, and `weight_mapping.csv`.

## Exact next work item

Run `python scripts/run_patch_ablation.py --run --max-runs 1` from the project root to execute the next pending case. The runner validates and exports the existing `P=4, seed=7` baseline first; then it trains `P=8, seed=7` for the frozen 20-epoch policy. If a run directory already exists but lacks a final summary, the runner adds `--resume` and continues from `last.pt`. It writes small per-run summaries, metrics, and `results/ablations/patch_size/progress.json`; inspect that file after each run for the exact next case. Once the first case passes, run `python scripts/run_patch_ablation.py --run` to complete the remaining serial matrix, or keep `--max-runs 1` for tighter checkpoints. The matrix is native 32×32 `P=4,8,16`, width 96, depth 4, heads 8, batch 128, AdamW, common augmentation, 20 epochs, and paired model/split seeds 7/11/19. The existing `P=4, seed=7` baseline is reused. The plan's larger width-192 study remains pending. Do not use the official test set while selecting variants. After all nine cases, publish the across-seed mean and sample standard deviation, training cost, uncertainty plot, and split caveat; update this file and push.

## Remaining milestone queue

1. Controlled patch/head/position/pooling/resolution experiments with recorded seeds and uncertainty; begin with patch size.
2. RetinaMNIST transfer study after the core checks.
3. Final documentation, README figures/tables, clean-checkout verification.

The baseline implementation is in commit `2311889`; the launch manifest records `d415e20` because that source was uncommitted at the instant training began, then committed without edits during the run. Use `git log -1 --oneline` for the latest status/results commit and `git status --short --branch` to verify that local `main` matches pushed `origin/main`. At every stopping point, update this file with passing checks, generated artifacts, and the exact next task, then commit and push.
