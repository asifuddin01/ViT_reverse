# Inside the Vision Transformer — A to Z execution guide

**Status:** implementation plan, prepared 2026-10-06. All phases are complete as of 2026-10-07: reference equivalence, CIFAR-10 baseline, attention diagnostics, CPU benchmark, three-seed patch-size, head-count, position, pooling, and width ablations, and the RetinaMNIST transfer study. The 128-pixel upsampled patch cost study and a width-192 patch matrix were not run; see `PROJECT_STATUS.md`.  
**Source brief:** the supplied “ViT Reverse Engineering Lab” project specification.  
**Working directory:** use this repository root as `vit-reverse-engineering-lab`; do not create a second Git repository inside it.

## 1. What the finished project must prove

1. A readable PyTorch ViT is implemented from elementary layers, including explicit Q/K/V and attention math.
2. A named, versioned pretrained ViT reference is inspected, its weights are mapped, and intermediate outputs are compared on identical inputs.
3. Attention and representation diagnostics can be generated without editing the model each time.
4. Runtime, memory, and parameter results use a documented protocol and real measurements.
5. Controlled experiments answer the patch, head, position, pooling, and resolution questions.
6. A separate retinal transfer study follows the core reverse-engineering result.
7. Tests, configuration, data provenance, figures, tables, and a report allow another engineer to reproduce the work.

The sequence is **understand → implement → verify → inspect → benchmark → experiment → report**. The reference-equivalence milestone is the central gate; do not present training accuracy alone as proof of reverse engineering.

## 2. Local starting point and practical scope

This workspace currently has no source files or commits. The host is macOS on Apple Silicon, has Python 3.12 available, and currently has no PyTorch, torchvision, timm, MedMNIST, or pytest installed. Use Python 3.12 for the project environment; the system `python3` is 3.14 and should not silently determine the dependency set. The pretrained reference checkpoint is about 346 MB, before caches and derived artifacts. [Checkpoint file](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k/blob/main/model.safetensors)

Three execution tiers keep the project workable on different hardware:

| Tier | Device | Required work | Practical limit |
|---|---|---|---|
| A | CPU | implementation, unit tests, one-sample FP32 equivalence, small smoke training | slow full training and benchmarks |
| B | Apple MPS | all core work, small-model experiments, device-specific benchmarks | report MPS memory as sampled current/driver allocation; do not label it CUDA peak memory |
| C | CUDA GPU | full experiment matrix and CUDA peak-memory benchmark | record GPU model, VRAM, CUDA and driver versions |

Start with batch size 1 for ViT-Base equivalence and batch size 32 or smaller for small-model training. Reduce batch size before changing the model definition if memory runs out. A nominal **8–10 week** schedule appears in §15; it is a planning estimate, not a claim about compute time.

## 3. Phase 0 — documentation discovery and reference contract

**Read these before writing the matching code:**

- Original ViT paper: [An Image Is Worth 16×16 Words](https://arxiv.org/abs/2010.11929).
- [timm model API](https://huggingface.co/docs/timm/reference/models), [preprocessing quickstart](https://huggingface.co/docs/timm/quickstart), [feature extraction](https://huggingface.co/docs/timm/feature_extraction).
- Reference source: [VisionTransformer and Block](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py), [Attention](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention.py), [fused-attention setting](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/config.py).
- [PyTorch module hooks and `load_state_dict`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html), and [floating-point limits](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

**Reference candidate:** `timm/vit_base_patch16_224.augreg_in21k_ft_in1k` at Hugging Face revision `b5b38b8f7dd93ce67d12e356a8771556b95c8cf9`. The model card identifies ImageNet-21k pretraining followed by ImageNet-1k fine-tuning. Its published `model.safetensors` SHA-256 is `c401d219603ac3e20b6373c7b198c78d3a733f80b755d148bda3bc320ae69800`; verify the downloaded artifact rather than assuming a cache entry matches. [Pinned model tree](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k/tree/b5b38b8f7dd93ce67d12e356a8771556b95c8cf9), [checkpoint record](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k/blob/main/model.safetensors)

**Allowed API starting points, to confirm against the installed pinned release:**

| Task | Documented entry point | Where to inspect |
|---|---|---|
| Instantiate reference | `timm.create_model(model_name, pretrained=True)` | timm model API and model card |
| Determine input transform | `timm.data.resolve_data_config(model.pretrained_cfg)` then `timm.data.create_transform(**data_cfg)` | timm quickstart |
| Inspect final token states | `model.forward_features(x)`; use `forward_head` for the classifier path | timm feature extraction |
| Capture module boundaries | `register_forward_hook`, then remove the returned handle | PyTorch `nn.Module` |
| Load mapped parameters | `load_state_dict(mapped, strict=True)` | PyTorch `nn.Module` |
| Assert numerical proximity | `torch.testing.assert_close(actual, expected, rtol=..., atol=...)` | PyTorch testing |
| Expose reference attention probabilities | Inspect `set_fused_attn(False)` or construct with `TIMM_FUSED_ATTN=0` **before** creating the reference; confirm behavior in the pinned timm source | timm attention/config source |

Create `docs/reference_contract.md` recording the **installed** Python/PyTorch/torchvision/timm versions, exact model ID and revision, checkpoint hash, printed `pretrained_cfg`, printed module tree, state-dict keys and shapes, `num_classes`, `global_pool`, `class_token`, norm epsilon, QKV bias, dropout, drop-path, and whether fused attention was enabled. Use observed values in implementation; the architecture in the brief is an initial target, not an excuse to assume every operation.

**Gate 0:** the reference ID/revision, source links, documentation-backed API list, and inspection checklist are written down. Instantiation and the inventory happen after the environment exists in Phase 1.

## 4. Phase 1 — repository and environment

Create this structure incrementally, rather than generating empty modules that are never used:

```text
README.md  PROJECT_SPEC.md  VIT_EXECUTION_PLAN.md  LICENSE
pyproject.toml  requirements-lock.txt  .gitignore
configs/{vit_base.yaml,vit_small.yaml,experiments/}
src/vit_lab/
  models/{patch_embedding.py,positional_embedding.py,attention.py,mlp.py,
          transformer_block.py,vision_transformer.py,classification_head.py}
  reference/{loader.py,hooks.py,weight_map.py,compare.py}
  analysis/{attention.py,activations.py,representations.py}
  benchmarking/{latency.py,memory.py,parameters.py,throughput.py}
  training/{data.py,trainer.py,metrics.py}
  utils/{config.py,seed.py,logging.py}
  cli.py
tests/
notebooks/
results/{figures,tables,checkpoints,logs}/
docs/{reference_contract.md,architecture.md,mathematics.md,
      reverse_engineering.md,benchmarking.md,experiments.md}
```

Copy the supplied brief into `PROJECT_SPEC.md` when implementation begins. Add `.venv/`, `data/`, checkpoints, downloaded weights, and bulky generated outputs to `.gitignore`; keep small CSV/JSON result summaries, figures, configs, and documentation versioned. Select the license deliberately and record dataset/checkpoint attribution separately from the code license.

**Bootstrap on this Mac:**

First write a minimal `pyproject.toml` so editable installation has a real package to install:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "vit-reverse-engineering-lab"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "torch", "torchvision", "timm", "medmnist", "numpy", "pandas",
  "matplotlib", "pyyaml", "scikit-learn", "pillow", "tqdm",
  "safetensors", "huggingface_hub",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]
```

Also create `src/vit_lab/__init__.py` before installation. Then:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip freeze > requirements-lock.txt
python -c 'import torch, torchvision, timm; print(torch.__version__, torchvision.__version__, timm.__version__); print("MPS:", torch.backends.mps.is_available())'
```

The `pip freeze` file captures the **tested local resolution**; do not invent version pins in advance. For a CUDA machine, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for its CUDA build, then resolve and record its own lock/environment file. Record `platform.platform()`, `torch.__version__`, and device details in every run. If the candidate timm version behaves differently from its docs, inspect the installed source and pin that exact version after tests pass.

Implement `inspect-reference` at this point, using the Phase 0 API list. It should instantiate the pinned reference, preprocess one RGB image using `pretrained_cfg` (a generated PIL test image is sufficient for this smoke check), run `eval()` inference, and save the module/parameter inventory to `results/logs/reference_inventory.json`.

**Gate 1:** editable import works; `pytest -q` starts; a CPU tensor and reference forward pass work; reference inventory and MPS availability are recorded. First Git commit contains the spec, plan, package skeleton, and environment instructions.

## 5. Phase 2 — math and tensor contract

Write `docs/mathematics.md` alongside code. For the reference shape, the expected flow is:

| Stage | Shape for batch `B` | Check |
|---|---|---|
| Image | `[B,3,224,224]` | `H` and `W` divisible by `P` unless padding is explicitly implemented |
| Patch projection, `P=16` | `[B,196,768]` | `196=(224/16)^2` |
| Prepend CLS and add position | `[B,197,768]` | one position per token |
| Q/K/V, `12` heads | each `[B,12,197,64]` | `768=12×64` |
| Attention probabilities | `[B,12,197,197]` | rows sum approximately to 1 in eval mode |
| Each block output | `[B,197,768]` | 12 blocks preserve shape |
| Final norm and CLS | `[B,197,768]`, `[B,768]` | inspect actual norm/pool order |
| Head | `[B,1000]` for chosen reference | class count from reference config |

Explain `Conv2d(C,D,kernel_size=P,stride=P)` as a shared linear projection on non-overlapping patches, and test it against an explicit `unfold` + `Linear` construction after copying equivalent weights. Explain `softmax(QKᵀ/√d_h)V`, pre-norm/residuals, per-token MLP, and why position must be added separately. A learned positional parameter usually has shape `[1,N+1,D]`; the **broadcast result** has batch shape `[B,N+1,D]`. Image resizing needs an explicit positional-embedding interpolation policy; avoid a silent shape fix.

**Gate 2:** the shape table and a hand-worked 2×2 patch example are documented; tests reject invalid `D % heads`, non-divisible image sizes, and mismatched positional length.

## 6. Phase 3 — implement the model in dependency order

Implement each component from the math and the observed reference contract; inspect the linked source for behavior but write original code. Keep every component testable.

| Order | Module | Implementation and immediate test |
|---|---|---|
| 1 | `patch_embedding.py` | efficient conv projection; optional explicit patch path for teaching; equality test after weight conversion |
| 2 | `positional_embedding.py` | learned positions; explicit CLS position; resolution policy; length assertions |
| 3 | `vision_transformer.py` CLS path | learned `[1,1,D]`, batch expand and concat; assert token order |
| 4 | `attention.py` | one packed `Linear(D,3D)` QKV projection or documented equivalent, reshape/transpose, scaled logits, softmax, dropout, value aggregation, output projection; optionally return probabilities |
| 5 | `mlp.py` | `Linear(D,4D) → GELU → Linear(4D,D)` with observed dropout |
| 6 | `transformer_block.py` | observed norm/residual/drop-path order; input/output shape and gradient checks |
| 7 | `classification_head.py` and full model | final norm/pool/head exactly as observed; `forward_features` and analysis return object |

Proposed API to implement:

```python
logits = model(images)
features = model.forward_features(images)
details = model(images, return_attention=True, return_hidden_states=True)
# details: logits, patch_embeddings, hidden_states, attention_maps, final_features
```

For ordinary training/inference, do not retain all attention matrices or hidden states. In analysis mode, document whether attention is **pre-dropout** or **post-dropout**; use pre-dropout probabilities for visualizations. Keep `return_attention=False` as the memory-efficient default. Do not hide the core calculation in `nn.MultiheadAttention` or `scaled_dot_product_attention` in the educational model.

**Tests:** patch shape/equality, QKV and map shapes, attention row sums, block shape, invalid configurations, variable batch size, deterministic eval, backward gradients through patch/attention/MLP/norm/head, and a one-batch optimizer step. Use `pytest -q` on CPU first. Do not infer correctness merely from shapes.

**Gate 3:** all tests pass and a tiny configuration overfits a fixed batch of 16–32 labeled examples, showing loss falls materially. This is a debugging check, not a reported accuracy result.

## 7. Phase 4 — small training run

Use [torchvision CIFAR-10](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CIFAR10.html) for the first real training run. Its images are 32×32. Use a small model at native resolution, for example `image_size=32, patch_size=4, embed_dim=192, depth=6, heads=8, mlp_ratio=4, classes=10`. These are **experiment defaults**, not properties of ViT-Base. Split only the official training set into a fixed stratified train/validation split; leave the official test set untouched until a model is selected. Save split indices and seed. Apply training augmentation only to training examples.

Implement a plain trainer with cross-entropy, AdamW, a documented learning-rate schedule, validation accuracy and loss, best-validation checkpoint, resume support, NaN checks, and a run manifest. Start with one epoch smoke test, then train a baseline until the loss curve is credible. A 20-epoch run is an initial budget; extend only with a recorded reason and use the same rule for compared variants. Record wall-clock time and device. Do not claim a tiny ViT trained from scratch on CIFAR-10 should match a large pretrained model.

A concrete pilot config is batch size 64 (or the largest that fits), AdamW learning rate `3e-4`, weight decay `0.05`, two warm-up epochs, cosine decay, 20 total epochs, and seed 7. This is a starting point to test, not a promised optimum. Freeze the final training protocol **before** the ablation matrix; if the pilot is still clearly improving, set one longer common budget for all variants. Compute normalization statistics from the training split or use a documented fixed CIFAR-10 convention; never estimate them from the test set.

**Gate 4:** one reproducible run completes, reloads its checkpoint, reproduces the same validation predictions, and writes `metrics.csv`, `config.yaml`, `environment.json`, and `checkpoint.pt` (the checkpoint may remain untracked).

## 8. Phase 5 — reference inventory and weight mapping

Use the pinned model in §3 and inspect the installed timm source. First create a reference with pretrained weights, `eval()`, FP32, dropout disabled by eval, a fixed one-image tensor, and exact preprocessing when using a real image. Save state-dict keys, shapes, module tree, and preprocessing configuration. For a direct function-equivalence check, feed **the same already-created tensor** to both models; preprocessing then cannot explain a difference.

Build `reference/weight_map.py` as an explicit mapping table. The **expected** key families below must be confirmed from the installed model inventory before use:

| Expected reference family | Our family | Shape or transformation to verify |
|---|---|---|
| `patch_embed.proj.{weight,bias}` | patch projection | conv shape `[D,3,P,P]` if both paths use Conv2d |
| `cls_token`, `pos_embed` | CLS, position | `[1,1,D]`, `[1,197,D]` |
| `blocks.i.norm1/2.*` | block norms | identical vector shape and epsilon |
| `blocks.i.attn.qkv.*` | QKV | packed `[3D,D]` and `[3D]`, or documented split |
| `blocks.i.attn.proj.*` | attention output | `[D,D]` and `[D]` |
| `blocks.i.mlp.fc1/fc2.*` | MLP | `[4D,D]` then `[D,4D]` |
| `norm.*`, `head.*` | final norm, classifier | confirm pool/head ordering and `num_classes` |

If splitting packed QKV, split weights and biases in Q,K,V order along output dimension. If converting Conv2d to an explicit Linear patch projector, document flatten order and reshape; prove equality with a unit test. Reject missing/unexpected keys for the same architecture with `strict=True`; do not quietly skip the classifier or load a partial dictionary while reporting full equivalence.

**Gate 5:** a machine-readable mapping report lists every parameter, source key, target key, source/target shape, transformation, and success/failure. Parameter counts and all mapped tensor values agree.

## 9. Phase 6 — layer-by-layer numerical equivalence

Register temporary hooks on matching **module boundaries**: patch projection, token-plus-position output (instrument our model and derive the reference point from its observed forward path), every block, final norm, final CLS/pooled feature, and logits. Ensure hooks are removed after each run. Also compare Q, K, V, attention probabilities, and attention output for the first block when debugging. The timm attention source can use fused attention, which may not expose probability matrices and may compute with a different kernel. For diagnostic comparisons, construct the reference with fused attention disabled using the API/env setting verified in Phase 0; run separate optimized benchmarks later. [timm attention source](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention.py), [fused setting](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/config.py)

For each stage save shape, dtype, max and mean absolute error, MSE, relative-error summary with a nonzero denominator floor, cosine similarity, and `assert_close` pass/fail. Start on CPU/FP32; compare MPS/CUDA only after CPU agreement. Begin with `rtol=1e-4, atol=1e-5` for local FP32 checks and document any adjusted threshold by stage/device with evidence. FP32 computations need not be bit-identical across kernels or devices. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

Debug at the **first divergent stage**: parameter mismatch → patch flattening → CLS/position order → norm epsilon/placement → QKV split/transposes/scaling → dropout/fusion → pool/head. Use at least three fixed inputs (random, zero, and one preprocessed real image). Do not accept “same top-1 class” as equivalence.

**Gate 6:** all expected intermediate stages and logits meet documented tolerances on CPU FP32, or each exception has an isolated, reproduced explanation. Produce `results/tables/equivalence.csv` and `docs/reverse_engineering.md` with the first-divergence debugging history.

## 10. Phase 7 — attention and representation analysis

For a selected image, layer, head, and sample, take **CLS-query to patch-key** probabilities, exclude CLS-to-CLS, reshape the patch vector to its true grid, upsample to input display size, and overlay it on the original image. Save the image ID, preprocessing, layer/head indices, and whether the map came from the reference or our model. Use a consistent color scale to compare heads/layers. Add patch-to-patch views only if the interpretation is clear. Label these as **attention diagnostics**, not explanations or lesion localization.

Save hidden states after blocks 1, 3, 6, 9, and 12 for ViT-Base. Compute CLS cosine similarity and layer-to-layer drift on a fixed image set; optionally run PCA on pooled features. Fit PCA only on the analysis set, not on hidden test labels. Store derived features or summaries, not every full attention tensor for an entire dataset.

**Gate 7:** a CLI command reproduces at least one early/middle/late layer panel and a representation-drift figure from a manifest without manual notebook edits.

## 11. Phase 8 — benchmarking and resolution study

Measure model parameters, trainable parameters, serialized/FP32 parameter bytes, latency for batch 1 and a larger batch, throughput, and device memory. State whether FLOPs are counted as one or two operations per multiply-add; report MACs separately if helpful. Validate analytical attention scaling with token count; do not label a formula as a measured hardware FLOP count.

Use `eval()` and `torch.inference_mode()`, fixed input shape/dtype, 20 warmups and at least 100 timed iterations as a starting protocol, repeated trials, and mean/median/std or percentiles. Time only model inference unless an end-to-end pipeline is explicitly named. For CUDA, synchronize or use CUDA events; asynchronous launches make naive wall-clock timing wrong. [PyTorch CUDA timing](https://docs.pytorch.org/docs/stable/notes/cuda). On MPS, call `torch.mps.synchronize()` before/after timings, and report `torch.mps.current_allocated_memory()` and `driver_allocated_memory()` as sampled values, noting their different coverage; these are not CUDA's `max_memory_allocated`/`max_memory_reserved`. [PyTorch MPS API](https://docs.pytorch.org/docs/stable/mps.html)

For fixed `P=16`, compare 224, 384, and 512 pixels: patch counts are **196, 576, 1024** and total tokens including CLS are **197, 577, 1025**. The attention matrix per head grows roughly with total-token squared; relative to 224 it is about **1×, 8.6×, 27.1×**. A full 512-pixel ViT-Base attention capture across 12 layers × 12 heads is roughly 605 MB for **one FP32 sample's maps alone**; capture selected layers/heads rather than all maps at once. Position embeddings need documented interpolation at non-reference resolutions. Benchmark forward passes only after confirming this adaptation is correct.

**Gate 8:** `results/tables/benchmark.csv` includes hardware, software, dtype, batch size, resolution, patch size, timing method, trials, measured latency/memory, and separately calculated MACs/FLOPs. Include plots of latency and sampled/peak memory versus tokens.

## 12. Phase 9 — controlled ablations

Use one immutable baseline config, one fixed data split, one optimizer/schedule/augmentation policy, and at least **three seeds** for accuracy comparisons when compute allows. Select epoch count and checkpoint rule before looking at test results; compare validation distributions and evaluate the test set once per selected configuration. Report mean and standard deviation across seeds, per-run values, and training cost. Record if compute constraints force one-seed exploratory results and do not treat them as strong conclusions.

| Question | Controlled variants | Primary outputs | Interpretation guard |
|---|---|---|---|
| Patch size on native 32×32 CIFAR-10 | `P=4,8,16` with same `D=192`, depth, heads, training policy | accuracy, token count, MACs, latency, memory | 32×32 with `P=32` yields one patch; exclude it from a meaningful spatial-detail comparison |
| Requested `P=8,16,32` cost study | 128×128 images from the **same** CIFAR-10 examples, same small model and training policy | token count, cost, exploratory accuracy | upsampling CIFAR-10 does not restore image detail; state this prominently |
| Head count | `4,8,12,16` at fixed `D=192`, `P`, depth | accuracy, latency, attention diversity | all divide 192; packed QKV parameter count is nearly unchanged |
| Position | learned vs zero/none; optional fixed sinusoidal | accuracy, convergence, feature behavior | use same initialization/training rule and record any differing parameter count |
| Pooling | CLS vs mean patch pooling | accuracy, convergence | define whether CLS token remains in the mean-pool model; prefer remove it and document the changed parameter count |
| Resolution | 224/384/512 at `P=16` for inference | tokens, latency, memory | interpolate positions and distinguish changed image content from pure scaling cost |

Use paired seed IDs and identical train/validation indices across variants. Preserve per-seed configs and logs. For optional attention diversity, define a numeric measure (for example, pairwise cosine similarity of head maps on fixed images) before reading the result; visual difference alone is anecdotal. Do not claim that head count or patch size has a universal accuracy direction.

**Gate 9:** every table row traces to a run manifest and artifact. Charts show uncertainty for accuracy results. The README table remains blank until measurements exist.

## 13. Phase 10 — retinal transfer extension

After Gate 6, use [MedMNIST RetinaMNIST](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py) at its published 224-pixel size for the portfolio extension. It contains 1,600 fundus images with official train/validation/test splits of 1,080/120/400, and the target is **five-level ordinal diabetic-retinopathy grading**. MedMNIST publishes 64/128/224 variants and licenses this subset CC BY 4.0; retain attribution and dataset version. [Dataset repository and license](https://github.com/MedMNIST/MedMNIST). The small validation set makes estimates noisy. This is research/education, not a clinical claim.

Start from the **verified reference-equivalent** pretrained backbone, replace the 1,000-class head with a five-output head, and fine-tune under a fixed protocol. Use the official splits; do not tune on test. Report balanced accuracy, macro-F1, confusion matrix, and an ordinal metric such as quadratic weighted kappa, with class counts and confidence intervals where feasible. Compare against a simple baseline (for example, majority class or a frozen-feature linear head) before claiming transfer benefit. Ensure one patient/eye cannot leak across custom splits if you make any; the official split should remain primary. Keep this experiment in `configs/experiments/retina.yaml` and `docs/experiments.md`, separate from the CIFAR ablations.

**Gate 10:** all metrics derive from saved predictions and labels; the best model was selected on validation data; the test set has a single documented final evaluation.

## 14. Phase 11 — documentation, portfolio presentation, and final verification

Write the docs while performing each phase; at the end, edit them into one coherent report:

- `README.md`: precise project claim, architecture figure, setup, reproducible commands, most important equivalence table, selected diagnostics, benchmark and ablation summary, limitations.
- `docs/architecture.md`: observed reference order and shape trace.
- `docs/mathematics.md`: derivations and toy examples.
- `docs/reference_contract.md` and `docs/reverse_engineering.md`: exact source/checkpoint/version, mapping, activation comparisons, discrepancies.
- `docs/benchmarking.md`: device, timing and memory protocol, FLOP convention.
- `docs/experiments.md`: hypotheses, controls, run IDs, results, uncertainty, interpretations, limitations.
- `PROJECT_SPEC.md`: original project brief, preserved for traceability.

Final verification checklist:

- [ ] Clean environment install from the committed instructions and dependency snapshot.
- [x] `pytest -q` passes, including gradients and reference equivalence.
- [x] Reference model ID, revision, hash, preprocessing, and installed package versions are recorded.
- [x] No unexpected/missing parameter keys for the direct reference comparison.
- [x] Every reported metric is computed from a saved run and has hardware/dtype/config metadata.
- [x] Every figure can be regenerated using a named command/config; no fabricated table cells.
- [x] Attention figures are described as diagnostics, not causal explanations.
- [x] Test data is not used for model or hyperparameter selection.
- [x] No raw dataset, downloaded checkpoint, secret, or huge tensor dump is committed.
- [ ] A second engineer can follow README commands from a clean checkout.

## 15. Suggested work schedule and stopping gates

| Week | Work | Evidence required before moving on |
|---|---|---|
| 1 | Phase 0–2: docs, environment, reference inventory, tensor contract | reference loads; architecture and transform captured |
| 2–3 | Phase 3: components and tests | all core tests + tiny-batch overfit pass |
| 4 | Phase 4: small-model training | logged CIFAR baseline, reloadable checkpoint |
| 5 | Phase 5–6: mapping and equivalence | strict weight load and per-layer comparison table |
| 6 | Phase 7–8: visual/representation tools and benchmarks | reproducible figures and measured benchmark CSV |
| 7–8 | Phase 9: controlled ablations | seeded runs and uncertainty-aware tables |
| 9 | Phase 10: retinal transfer | held-out test report from saved predictions |
| 10 | Phase 11: final verification and README | clean-run reproduction and complete report |

**Minimum portfolio milestone if compute is limited:** complete Gates 0–8, one carefully controlled ablation with three seeds, and the report. The full supplied brief's definition of done also requires all listed ablations and retinal transfer, so identify unfinished rows honestly if stopping earlier.

## 16. Planned command interface

These are **commands to implement during the phases**, not commands that work in the currently empty repository. Keep the interface stable so the README is executable:

```bash
python -m vit_lab.cli inspect-reference --config configs/vit_base.yaml
python -m vit_lab.cli train --config configs/vit_small.yaml --seed 7
python -m vit_lab.cli compare --config configs/vit_base.yaml --device cpu
python -m vit_lab.cli visualize --config configs/vit_base.yaml --image path/to/image.jpg --layer 11 --head 0
python -m vit_lab.cli benchmark --config configs/vit_base.yaml --device mps --batch-size 1
python -m vit_lab.cli ablate --config configs/experiments/patch_size.yaml
python -m vit_lab.cli train --config configs/experiments/retina.yaml --seed 7
pytest -q
```

Each command should write a timestamped run directory with `config.yaml`, `environment.json`, `metrics.json` or `metrics.csv`, logs, and any small figures. The run manifest must include seed, dataset/split, model config, preprocessing, optimizer/scheduler, batch size, epochs, precision, device, versions, and Git commit. A failed run should retain its error and config instead of being silently overwritten.

## 17. Decisions that prevent common project failures

1. **Reference first, then adaptation.** Direct equivalence uses the exact 224-pixel, 1,000-class pretrained architecture. CIFAR and retina heads/resolutions are separate experiments.
2. **Keep the reference pinned.** A model name alone can resolve to different default weights later; record the Hub revision and artifact hash.
3. **Use the reference transform.** Do not assume ImageNet mean/std; read `pretrained_cfg` for the selected checkpoint. [timm quickstart](https://huggingface.co/docs/timm/quickstart)
4. **Separate diagnostic and optimized attention.** Manual attention is needed for inspectability; fused kernels are a separate performance comparison.
5. **Treat numerical tolerance as measured evidence.** Floating-point results vary with kernels and hardware; save errors at every stage, not just one pass/fail flag.
6. **Do not over-interpret resized low-resolution data.** Token/computation scaling is real; added image detail is not.
7. **State exactly what is measured.** Parameter bytes, sampled MPS memory, CUDA peak memory, analytical MACs, and wall-clock latency are different quantities.
8. **Document actual negative findings.** If an ablation changes little or hurts performance, that is a valid result when controls and uncertainty are clear.

## 18. Primary source index

- [ViT paper](https://arxiv.org/abs/2010.11929) and [How to train your ViT?](https://arxiv.org/abs/2106.10270).
- [Pinned timm model](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k/tree/b5b38b8f7dd93ce67d12e356a8771556b95c8cf9).
- [timm model API](https://huggingface.co/docs/timm/reference/models), [quickstart and preprocessing](https://huggingface.co/docs/timm/quickstart), [feature extraction](https://huggingface.co/docs/timm/feature_extraction).
- [timm VisionTransformer source](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py), [attention source](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention.py), [layer config](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/config.py).
- [PyTorch module API](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [testing API](https://docs.pytorch.org/docs/stable/testing.html), [numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html), [CUDA timing](https://docs.pytorch.org/docs/stable/notes/cuda), [MPS memory](https://docs.pytorch.org/docs/stable/mps.html).
- [torchvision CIFAR-10](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CIFAR10.html), [MedMNIST data definitions](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py), [MedMNIST license](https://github.com/MedMNIST/MedMNIST).
