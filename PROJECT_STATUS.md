# Project handoff status

**Updated:** 2026-10-07  
**Project directory:** `/Users/mdasifuddin/AI/ViT_reverse`  
**Branch:** `main`  
**Remote:** `https://github.com/asifuddin01/ViT_reverse.git`  
**Current task:** Implement reproducible parameter, MAC/FLOP, latency, throughput, and CPU memory benchmarks for the educational ViT and pinned reference.

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

## Current limits and honest interpretation

- The real-image comparison uses a 32×32 CIFAR-10 photograph resized for ViT-Base. It verifies numerical behavior, not native high-resolution detail.
- `torch.backends.mps.is_available()` returned `False` on this host. CPU is the verified device; CUDA and MPS results are unmeasured.
- The CIFAR result has one seed and one validation split. There is no CIFAR test accuracy yet, no uncertainty estimate across training seeds, and no medical transfer, latency/memory benchmark, or ablation result. Do not claim their results.
- The attention figures use 32×32 photographs enlarged to 224×224 and are diagnostics, not explanations or localization maps. The representation summary covers only four selected images.
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
git status --short --branch
git remote -v
```

If `.venv` is missing, recreate it with Python 3.12 and `python -m pip install -e '.[dev]'`. The first `inspect-reference` without `--offline` downloads the pinned checkpoint. The comparison creates `results/tables/equivalence.csv`, `equivalence.json`, and `weight_mapping.csv`.

## Exact next work item

Implement the Phase 8 benchmark CLI and methodology. Record model and trainable parameter counts, FP32 serialized parameter bytes, analytical MACs (with an explicit FLOP convention), measured CPU inference latency/throughput for batch 1 and a larger batch, and process memory with a clearly named measurement method. Benchmark ViT-Base at 224 pixels and a resolution study at 384 and 512 pixels using documented positional-embedding interpolation. Separate measured hardware values from analytical formulas and avoid capturing all attention maps at high resolution. Save reproducible CSV and metadata, validate analytical counts on a small model, document hardware/software/trials/warmup/timing, then update this file, commit, and push before ablations.

## Remaining milestone queue

1. Parameter, MAC/FLOP, latency, throughput, and memory benchmarks.
2. Controlled patch/head/position/pooling/resolution experiments with recorded seeds and uncertainty.
3. RetinaMNIST transfer study after the core checks.
4. Final documentation, README figures/tables, clean-checkout verification.

The baseline implementation is in commit `2311889`; the launch manifest records `d415e20` because that source was uncommitted at the instant training began, then committed without edits during the run. Use `git log -1 --oneline` for the latest status/results commit and `git status --short --branch` to verify that local `main` matches pushed `origin/main`. At every stopping point, update this file with passing checks, generated artifacts, and the exact next task, then commit and push.
