# Project handoff status

**Updated:** 2026-10-07  
**Project directory:** `/Users/mdasifuddin/AI/ViT_reverse`  
**Branch:** `main`  
**Remote:** `https://github.com/asifuddin01/ViT_reverse.git`  
**Current task:** Complete the full seed-7 CIFAR-10 CPU baseline and verify its saved best checkpoint.

## Completed, in order

1. Repository and Python 3.12 package setup. The original brief is `PROJECT_SPEC.md`; the work plan is `VIT_EXECUTION_PLAN.md`. Dependencies are captured in `requirements-lock.txt`.
2. Pinned timm ViT-Base/16 reference inspected and its `model.safetensors` SHA-256 verified. The checkpoint has 152 tensors and 86,567,656 parameters. See `docs/reference_contract.md` and `vit-lab inspect-reference`.
3. From-scratch patch projection and full inspectable ViT implemented. Patch convolution equals explicit patch extraction in values and input gradients. The network-free timm oracle test passes.
4. The reference's 152 parameters map directly to our model with identical names and shapes. A CPU/FP32 comparison with reference fused attention disabled passed **100 of 100** captured stage comparisons (25 stages × seeded random, zeros, generated RGB, and an official CIFAR-10 image), with recorded maximum absolute difference `0.0` for those inputs. See `docs/reverse_engineering.md` and `results/tables/`.
5. The user-provided GitHub repository is configured as `origin`; `main` was pushed and tracks `origin/main`.
6. Official CIFAR-10 data downloaded and passed torchvision's integrity check. The deterministic stratified train/validation loader, AdamW trainer, checkpoint/resume logic, and CLI are implemented. Two four-batch smoke runs passed, including exact validation-prediction reproduction after checkpoint reload.
7. `pytest -q` passed **18 tests** after the latest trainer changes.

## Current limits and honest interpretation

- The real-image comparison uses a 32×32 CIFAR-10 photograph resized for ViT-Base. It verifies numerical behavior, not native high-resolution detail.
- `torch.backends.mps.is_available()` returned `False` on this host. CPU is the verified device; CUDA and MPS results are unmeasured.
- A full CIFAR-10 seed-7 CPU run was started in `results/runs/cifar_tiny_seed7`. Its final accuracy is not yet known. No medical transfer, latency/memory benchmark, attention figure, or ablation has been run. Do not claim their results.
- Model checkpoints and datasets are intentionally excluded from Git. The pinned pretrained checkpoint is in the Hugging Face cache and can be re-downloaded from the configured revision.

## Resume commands

```bash
cd /Users/mdasifuddin/AI/ViT_reverse
source .venv/bin/activate
pytest -q
vit-lab inspect-reference --config configs/vit_base.yaml --offline
vit-lab compare --config configs/vit_base.yaml --offline --device cpu
vit-lab train --config configs/vit_tiny_cifar.yaml --run-dir results/runs/cifar_tiny_seed7 --device cpu --resume
git status --short --branch
git remote -v
```

If `.venv` is missing, recreate it with Python 3.12 and `python -m pip install -e '.[dev]'`. The first `inspect-reference` without `--offline` downloads the pinned checkpoint. The comparison creates `results/tables/equivalence.csv`, `equivalence.json`, and `weight_mapping.csv`.

## Exact next work item

Inspect `results/runs/cifar_tiny_seed7/metrics.csv` and `last.pt` to see whether the active 20-epoch CPU run completed. If it stopped after any finished epoch, resume with the command above. Do **not** start a second run in that directory without `--resume`. After completion, verify `summary.json`, checkpoint reload, and the loss/accuracy curves; add the measured result to `docs/experiments.md`, commit a small result summary (not checkpoints), and push `main`. The run's launch manifest may show the previous Git commit because source changes were committed immediately after launch; the source contents were unchanged during the run. Explain this provenance detail in the final report.

## Remaining milestone queue

1. CIFAR-10 training baseline and checkpoint reload proof.
2. Real-image equivalence input and interpretation update.
3. Attention and representation diagnostics.
4. Parameter, MAC/FLOP, latency, throughput, and memory benchmarks.
5. Controlled patch/head/position/pooling/resolution experiments with recorded seeds and uncertainty.
6. RetinaMNIST transfer study after the core checks.
7. Final documentation, README figures/tables, clean-checkout verification.

At every stopping point, update this file with the latest passing checks, generated artifacts, exact next task, Git commit, and remote push state.
