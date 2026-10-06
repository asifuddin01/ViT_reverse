# Reverse-engineering and numerical verification

## Reference and environment

The reference is [`timm/vit_base_patch16_224.augreg_in21k_ft_in1k`](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k/tree/b5b38b8f7dd93ce67d12e356a8771556b95c8cf9) at revision `b5b38b8f7dd93ce67d12e356a8771556b95c8cf9`. The downloaded `model.safetensors` SHA-256 was checked against `c401d219603ac3e20b6373c7b198c78d3a733f80b755d148bda3bc320ae69800` before loading. The run used Python 3.12, PyTorch 2.14.1, torchvision 0.29.1, and timm 1.0.30 on CPU/FP32. See [reference_contract.md](reference_contract.md) for preprocessing and observed architecture details.

## Inspection and implementation path

1. Read the installed timm `VisionTransformer`, `Block`, `Attention`, and `Mlp` source and recorded the module tree and state-dict shapes with `vit-lab inspect-reference`.
2. Implemented patch projection, CLS and learned positions, packed QKV, explicit softmax attention, per-token MLP, pre-norm residual blocks, final norm, CLS pooling, and the classifier in `src/vit_lab/models/`.
3. Kept the independently written parameter names aligned with the observed reference names. `load_exact_reference_weights` verifies all keys and shapes, then uses `strict=True`. The [weight map](../results/tables/weight_mapping.csv) contains **152 identity mappings**; no tensor reshape, transpose, or omission was needed. This is an explicit result of the chosen tensor layouts, not an assumed automatic conversion.
4. Disabled reference fused attention **at construction** for the diagnostic run. Both models then perform the explicit Q/K/V, scaled matrix multiplication, softmax, value aggregation, and output projection. Optimized-kernel performance is a separate benchmark question.
5. Registered temporary forward hooks at matching module boundaries and removed every hook after inference. The complete [equivalence table](../results/tables/equivalence.csv) and [run metadata](../results/tables/equivalence.json) are committed as small reproducible artifacts.

The input transform is resolved from the pinned checkpoint config, not from generic ImageNet defaults. For function equivalence, the **same already-created input tensor** is passed to both models; a preprocessing mismatch therefore cannot hide inside the model comparison.

## Compared stages and result

Three fixed inputs were used: a seeded random tensor, an all-zero tensor, and a generated RGB image passed through the reference preprocessing. All are synthetic. Each input produced 25 captured comparisons: input, patch embedding, token-plus-position output, first-block norm/QKV/attention probabilities/projection/MLP sublayers, all 12 block outputs, final norm, pooled CLS features, and logits.

| Measure | Observed result |
|---|---:|
| State-dict keys and shapes mapped | 152 / 152 |
| Stage comparisons meeting `rtol=1e-4`, `atol=1e-5` | 75 / 75 |
| Largest absolute difference across captured tensors | 0.0 on these CPU/FP32 inputs |
| Largest absolute logit difference | 0.0 on these CPU/FP32 inputs |

The comparator also stores mean absolute error, MSE, relative error with a denominator floor, and flattened cosine similarity. Exact equality here is an **observed property of this pinned software/device/configuration and these inputs**. It does not imply bitwise equivalence across devices, fused kernels, precision modes, or future library versions. A real photographed input and non-CPU comparisons remain to be added when data and hardware are available.

## Reproduce

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
vit-lab inspect-reference --config configs/vit_base.yaml
vit-lab compare --config configs/vit_base.yaml --offline --device cpu
pytest -q
```

The first command using the Hub downloads the pinned checkpoint into the local cache. The comparison command uses `--offline` after that. Add `--image path/to/photo.jpg` to compare a real image with the checkpoint's preprocessing. The comparator exits with status 1 if any recorded stage misses the tolerance.
