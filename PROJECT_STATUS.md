# Project handoff status

**Updated:** 2026-10-07  
**Project directory:** `/Users/mdasifuddin/AI/ViT_reverse`  
**Branch:** `main`  
**Remote:** `https://github.com/asifuddin01/ViT_reverse.git`  
**Current task:** Implement attention and hidden-representation diagnostics on fixed real images, starting with CLS-to-patch attention maps. The CIFAR-10 baseline is complete.

## Completed, in order

1. Repository and Python 3.12 package setup. The original brief is `PROJECT_SPEC.md`; the work plan is `VIT_EXECUTION_PLAN.md`. Dependencies are captured in `requirements-lock.txt`.
2. Pinned timm ViT-Base/16 reference inspected and its `model.safetensors` SHA-256 verified. The checkpoint has 152 tensors and 86,567,656 parameters. See `docs/reference_contract.md` and `vit-lab inspect-reference`.
3. From-scratch patch projection and full inspectable ViT implemented. Patch convolution equals explicit patch extraction in values and input gradients. The network-free timm oracle test passes.
4. The reference's 152 parameters map directly to our model with identical names and shapes. A CPU/FP32 comparison with reference fused attention disabled passed **100 of 100** captured stage comparisons (25 stages × seeded random, zeros, generated RGB, and an official CIFAR-10 image), with recorded maximum absolute difference `0.0` for those inputs. See `docs/reverse_engineering.md` and `results/tables/`.
5. The user-provided GitHub repository is configured as `origin`; `main` was pushed and tracks `origin/main`.
6. Official CIFAR-10 data downloaded and passed torchvision's integrity check. The deterministic stratified train/validation loader, AdamW trainer, checkpoint/resume logic, and CLI are implemented. Two four-batch smoke runs passed, including exact validation-prediction reproduction after checkpoint reload.
7. `pytest -q` passed **18 tests** after the latest trainer changes.
8. The full 20-epoch seed-7 CIFAR-10 CPU baseline completed. Best validation accuracy was **69.50% at epoch 20**; the official test split was untouched. The best checkpoint reproduced its saved validation logits exactly after reload. Versioned results are `results/tables/cifar_tiny_seed7_metrics.csv`, `results/tables/cifar_tiny_seed7_summary.json`, and `results/figures/cifar_tiny_seed7_training.svg`. See `docs/experiments.md`.

## Current limits and honest interpretation

- The real-image comparison uses a 32×32 CIFAR-10 photograph resized for ViT-Base. It verifies numerical behavior, not native high-resolution detail.
- `torch.backends.mps.is_available()` returned `False` on this host. CPU is the verified device; CUDA and MPS results are unmeasured.
- The CIFAR result has one seed and one validation split. There is no CIFAR test accuracy yet, no uncertainty estimate across training seeds, and no medical transfer, latency/memory benchmark, attention figure, or ablation result. Do not claim their results.
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
git status --short --branch
git remote -v
```

If `.venv` is missing, recreate it with Python 3.12 and `python -m pip install -e '.[dev]'`. The first `inspect-reference` without `--offline` downloads the pinned checkpoint. The comparison creates `results/tables/equivalence.csv`, `equivalence.json`, and `weight_mapping.csv`.

## Exact next work item

Implement a reproducible CLI analysis command for a small fixed set of real images, starting with an official CIFAR-10 image already in `data/cifar_train_00000.png`. Load the pinned ViT-Base checkpoint into the educational model, request pre-dropout attention probabilities and hidden states, and save only selected layers/heads to avoid retaining all maps. Produce a CLS-query-to-patch-key overlay labeled with image ID, preprocessing, layer/head, model/checkpoint identity, and the warning that attention is a diagnostic rather than an explanation. Validate map dimensions, row sums, and deterministic repeatability. Then compute fixed-set CLS cosine similarity and layer-to-layer drift after blocks 1, 3, 6, 9, and 12. Put versioned small figures/tables in `results/`, document the method, run tests, commit, push, and update this file before the benchmark task.

## Remaining milestone queue

1. Attention and representation diagnostics.
2. Parameter, MAC/FLOP, latency, throughput, and memory benchmarks.
3. Controlled patch/head/position/pooling/resolution experiments with recorded seeds and uncertainty.
4. RetinaMNIST transfer study after the core checks.
5. Final documentation, README figures/tables, clean-checkout verification.

The baseline implementation is in commit `2311889`; the launch manifest records `d415e20` because that source was uncommitted at the instant training began, then committed without edits during the run. Use `git log -1 --oneline` for the latest status/results commit and `git status --short --branch` to verify that local `main` matches pushed `origin/main`. At every stopping point, update this file with passing checks, generated artifacts, and the exact next task, then commit and push.
