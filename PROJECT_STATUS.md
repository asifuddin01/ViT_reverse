# Project handoff status

**Updated:** 2026-10-06  
**Project directory:** `/Users/mdasifuddin/AI/ViT_reverse`  
**Branch:** `main`  
**Remote:** `https://github.com/asifuddin01/ViT_reverse.git`  
**Current task:** Publish the completed core work to the user-provided GitHub repository, then begin the small CIFAR-10 training task.

## Completed, in order

1. Repository and Python 3.12 package setup. The original brief is `PROJECT_SPEC.md`; the work plan is `VIT_EXECUTION_PLAN.md`. Dependencies are captured in `requirements-lock.txt`.
2. Pinned timm ViT-Base/16 reference inspected and its `model.safetensors` SHA-256 verified. The checkpoint has 152 tensors and 86,567,656 parameters. See `docs/reference_contract.md` and `vit-lab inspect-reference`.
3. From-scratch patch projection and full inspectable ViT implemented. Patch convolution equals explicit patch extraction in values and input gradients. The network-free timm oracle test passes.
4. The reference's 152 parameters map directly to our model with identical names and shapes. A CPU/FP32 comparison with reference fused attention disabled passed **75 of 75** captured stage comparisons (25 stages × 3 synthetic inputs), with recorded maximum absolute difference `0.0` for those inputs. See `docs/reverse_engineering.md` and `results/tables/`.
5. `pytest -q` passed **15 tests** after the latest model and comparator changes.

## Current limits and honest interpretation

- The three comparison inputs are seeded random pixels, zeros, and a generated RGB image with checkpoint preprocessing. No real photograph has yet been compared.
- `torch.backends.mps.is_available()` returned `False` on this host. CPU is the verified device; CUDA and MPS results are unmeasured.
- No CIFAR training, medical transfer, latency/memory benchmark, attention figure, or ablation has been run. Do not claim their results.
- Model checkpoints and datasets are intentionally excluded from Git. The pinned pretrained checkpoint is in the Hugging Face cache and can be re-downloaded from the configured revision.

## Resume commands

```bash
cd /Users/mdasifuddin/AI/ViT_reverse
source .venv/bin/activate
pytest -q
vit-lab inspect-reference --config configs/vit_base.yaml --offline
vit-lab compare --config configs/vit_base.yaml --offline --device cpu
git status --short --branch
git remote -v
```

If `.venv` is missing, recreate it with Python 3.12 and `python -m pip install -e '.[dev]'`. The first `inspect-reference` without `--offline` downloads the pinned checkpoint. The comparison creates `results/tables/equivalence.csv`, `equivalence.json`, and `weight_mapping.csv`.

## Exact next work item

Connect and push `main` to `origin` after confirming the remote is empty or reconciling any remote commits. Then implement the Phase 4 CIFAR-10 data/training pipeline from `VIT_EXECUTION_PLAN.md`: fixed stratified train/validation indices from the official training split, one-epoch smoke run, a reloadable best-validation checkpoint, metrics/environment manifest, and one real-image reference comparison using a dataset example. Keep test data out of model selection. Run tests and update this file with the completed evidence before moving to attention visualization or benchmarks.

## Remaining milestone queue

1. CIFAR-10 training baseline and checkpoint reload proof.
2. Real-image equivalence input and interpretation update.
3. Attention and representation diagnostics.
4. Parameter, MAC/FLOP, latency, throughput, and memory benchmarks.
5. Controlled patch/head/position/pooling/resolution experiments with recorded seeds and uncertainty.
6. RetinaMNIST transfer study after the core checks.
7. Final documentation, README figures/tables, clean-checkout verification.

At every stopping point, update this file with the latest passing checks, generated artifacts, exact next task, Git commit, and remote push state.
