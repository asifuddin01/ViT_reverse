# Project handoff status

**Updated:** 2026-10-07  
**Project directory:** `/Users/mdasifuddin/AI/ViT_reverse`  
**Branch:** `main`  
**Remote:** `https://github.com/asifuddin01/ViT_reverse.git`  
**Current task:** Run the controlled head-count ablation. Eleven of twelve cases completed; the final case is `heads=16, seed=19`.

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
11. The patch-size ablation's nine configs and resumable serial runner were prepared before new training. The frozen protocol is documented in `docs/experiments.md`. `pytest -q -p no:cacheprovider` passed **23 tests**, including config invariance and freeze checks.
12. The existing `P=4, seed=7` run was validated and copied into versioned `results/ablations/patch_size/` as a small summary and per-epoch metrics. Its full split and checkpoint remain ignored under `results/runs/cifar_tiny_seed7`. The new `P=8, seed=7` run began from protocol commit `abbb32d`.
13. The full `P=8, seed=7` run completed with best validation accuracy **62.96% at epoch 19** and exact validation-prediction reproduction after checkpoint reload. It used the same saved train/validation split hash as `P=4, seed=7`. Its 20 epochs took 614.0 seconds. Small evidence is in `results/ablations/patch_size/patch_p08_seed7_*`.
14. The full `P=16, seed=7` run completed with best validation accuracy **54.98% at epoch 20**, exact checkpoint reload, and 436.4 seconds of epoch time. Its split hash matches both other seed-7 variants. Small evidence is in `results/ablations/patch_size/patch_p16_seed7_*`. The seed-7 values are paired observations, not a multi-seed estimate.
15. The full `P=4, seed=11` run completed with best validation accuracy **72.84% at epoch 20**, exact checkpoint reload, and 1,393.9 seconds of epoch time. Its config, split hash, environment, and all 20 epochs are in `results/ablations/patch_size/patch_p04_seed11_*`. Its split is distinct from seed 7 and paired with the other seed-11 variants.
16. The full `P=8, seed=11` run completed with best validation accuracy **64.70% at epoch 19**, exact checkpoint reload, and 611.6 seconds of epoch time. Its split hash matches `P=4, seed=11`. Small evidence is in `results/ablations/patch_size/patch_p08_seed11_*`.
17. The full `P=16, seed=11` run completed with best validation accuracy **56.46% at epoch 20**, exact checkpoint reload, and 436.4 seconds of epoch time. Its split hash matches the other seed-11 variants. Small evidence is in `results/ablations/patch_size/patch_p16_seed11_*`.
18. The full `P=4, seed=19` run completed with best validation accuracy **72.42% at epoch 20**, exact checkpoint reload, and 1,393.4 seconds of epoch time. Small evidence is in `results/ablations/patch_size/patch_p04_seed19_*`.
19. The full `P=8, seed=19` run completed with best validation accuracy **62.54% at epoch 19**, exact checkpoint reload, and 615.2 seconds of epoch time. Its split hash matches `P=4, seed=19`. Small evidence is in `results/ablations/patch_size/patch_p08_seed19_*`.
20. The final `P=16, seed=19` run completed with best validation accuracy **56.16% at epoch 20** and exact checkpoint reload. The nine-run patch-size matrix is complete: mean ± sample SD across three paired seeds was **71.59 ± 1.82% (P=4)**, **63.40 ± 1.15% (P=8)**, and **55.87 ± 0.78% (P=16)**. Every run has 20 epoch rows; split hashes match within each seed. `results/ablations/patch_size/aggregate.csv`, `cost.csv/.json`, and `results/figures/patch_size_ablation.svg/.png` hold the small result artifacts. The aggregate, cost trials, and figures were independently checked; `pytest -q -p no:cacheprovider` passed **23 tests** and `pip check` found no broken requirements. See `docs/experiments.md` for per-seed values, training time, measured local CPU inference cost, and limits.
21. The twelve-case head-count protocol was frozen before new training: `heads=4,8,12,16`, paired seeds 7/11/19, width 96, patch size 8, and all previous P=8 training controls. Nine new configs live in `configs/experiments/head_count/`; the three completed eight-head runs are reused. `src/vit_lab/experiments/head_count.py` is the serial resume/export runner. `pytest -q -p no:cacheprovider` passed **24 tests**, including config invariance and freeze checks.
22. The full `heads=4, seed=7` run completed with best validation accuracy **63.08% at epoch 19**, exact checkpoint reload, and 516.7 seconds of epoch time. Its split hash matches reused `heads=8, seed=7` (**62.96%**). Small evidence for both and the reused heads-8 seed-11/19 runs is in `results/ablations/head_count/`.
23. The full `heads=12, seed=7` run completed with best validation accuracy **62.72% at epoch 19**, exact checkpoint reload, and 790.5 seconds of epoch time. Its split hash matches the four- and eight-head seed-7 cases.
24. The full `heads=16, seed=7` run completed with best validation accuracy **63.10% at epoch 19**, exact checkpoint reload, and 883.8 seconds of epoch time. Its split hash matches the other three seed-7 cases.
25. The full `heads=4, seed=11` run completed with best validation accuracy **64.70% at epoch 20**, exact checkpoint reload, and 545.4 seconds of epoch time. Its split hash matches reused `heads=8, seed=11`, which also reached **64.70%**.
26. The full `heads=12, seed=11` run completed with best validation accuracy **63.62% at epoch 19**, exact checkpoint reload, and 679.1 seconds of epoch time. Its split hash matches the other seed-11 cases.
27. The full `heads=16, seed=11` run completed with best validation accuracy **63.54% at epoch 20**, exact checkpoint reload, and 815.6 seconds of epoch time. Its split hash matches the other seed-11 cases.
28. The full `heads=4, seed=19` run completed with best validation accuracy **63.50% at epoch 20**, exact checkpoint reload, and 457.0 seconds of epoch time. Its split hash matches reused `heads=8, seed=19` (**62.54%**).
29. The full `heads=12, seed=19` run completed with best validation accuracy **62.54% at epoch 20**, exact checkpoint reload, and 631.1 seconds of epoch time. Its split hash matches the other seed-19 cases. `results/ablations/head_count/progress.json` names `heads=16, seed=19` next. Aggregate interpretation waits for this final case.

## Current limits and honest interpretation

- The real-image comparison uses a 32×32 CIFAR-10 photograph resized for ViT-Base. It verifies numerical behavior, not native high-resolution detail.
- `torch.backends.mps.is_available()` returned `False` on this host. CPU is the verified device; CUDA and MPS results are unmeasured.
- The initial CIFAR baseline alone has one seed. The patch-size ablation has three paired seeds and reports descriptive sample SD; different seed IDs use different validation splits. There is no CIFAR official test accuracy or medical transfer result. The head-count ablation is in progress; position and pooling ablations remain unmeasured.
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
python scripts/run_patch_ablation.py --run
cat results/ablations/patch_size/progress.json
python scripts/plot_patch_ablation.py results/ablations/patch_size/aggregate.csv results/figures/patch_size_ablation.svg
python scripts/benchmark_patch_ablation.py --warmups 10 --trials 50 --threads 4
python scripts/run_head_ablation.py --prepare-only
python scripts/run_head_ablation.py --run --max-runs 1
git status --short --branch
git remote -v
```

If `.venv` is missing, recreate it with Python 3.12 and `python -m pip install -e '.[dev,experiments]'`. The first `inspect-reference` without `--offline` downloads the pinned checkpoint. The comparison creates `results/tables/equivalence.csv`, `equivalence.json`, and `weight_mapping.csv`. `progress.json` now has no pending patch-size cases.

## Exact next work item

Inspect `results/ablations/head_count/progress.json` for the next pending case. Execute `python scripts/run_head_ablation.py --run --max-runs 1` from the project root for one case, or `--run` for the remaining serial matrix. If an incomplete run directory exists, the runner adds `--resume` and continues from `last.pt`; never start a second non-resume run there. The runner checks paired split hashes and checkpoint reloads and exports small evidence after each completed run. After all twelve cases, publish mean/sample SD and local inference cost, then update this file and push. Do not use the official test set to choose a variant. The larger width-192 study remains pending.

## Remaining milestone queue

1. Controlled head-count, position, pooling, and larger-width experiments with recorded seeds and uncertainty; begin with head count.
2. RetinaMNIST transfer study after the core checks.
3. Final documentation, README figures/tables, clean-checkout verification.

The baseline implementation is in commit `2311889`; the launch manifest records `d415e20` because that source was uncommitted at the instant training began, then committed without edits during the run. Use `git log -1 --oneline` for the latest status/results commit and `git status --short --branch` to verify that local `main` matches pushed `origin/main`. At every stopping point, update this file with passing checks, generated artifacts, and the exact next task, then commit and push.
