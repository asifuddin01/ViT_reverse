# Attention and representation diagnostics

## Reproduce

Use the official CIFAR-10 training files in `data/` and the verified pinned checkpoint in the Hugging Face cache. The training command downloads CIFAR-10; alternatively, fetch it directly with `CIFAR10(root="data", train=True, download=True)` from torchvision. From the project root:

```bash
vit-lab analyze --config configs/vit_base.yaml --data-root data \
  --output-dir results/analysis/cifar10_seed7 --indices 0 1 2 3 \
  --attention-layers 1 6 12 --heads 0 1 \
  --hidden-layers 1 3 6 9 12 --device cpu --offline
```

Omit `--offline` to fetch the pinned Hugging Face files if they are not cached. The command hashes the checkpoint and loads its 152 tensors into the independently coded ViT with exact key and shape checks. It uses the checkpoint's bicubic, center-crop, 224-pixel transform and `(0.5, 0.5, 0.5)` channel mean and standard deviation. These are 32×32 CIFAR-10 photographs enlarged for the pretrained model; fine spatial detail is absent.

## What is captured

The first image, `cifar10_train_00000`, supplies six attention figures: layers 1, 6, and 12 (one-based), heads 0 and 1 (zero-based). Each map is the **CLS query's pre-dropout softmax probability over patch keys**, excluding the CLS key. The hook copies only that row from the selected heads; it does not save every attention matrix. The 196 patch values form a 14×14 grid and are bilinearly enlarged over the exact preprocessed image. All six overlays share one absolute color scale from 0 to 0.10917. The matching full-row sums are within 0.000001 of one; a repeated first-image inference produced bitwise identical logits and captured rows on CPU.

For CIFAR-10 training indices 0, 1, 2, and 3, the command saves post-block CLS vectors after blocks 1, 3, 6, 9, and 12. It computes cosine similarity to the final selected vector and `1 − cosine` from the previous **selected** block. These are vector geometry diagnostics, not a measure of semantic understanding. No labels or test images enter the calculations.

## Observed output

For image 0, layer 1/head 0 assigns **99.54%** of the CLS row to the CLS key and only **0.46%** to patches; its patch overlay is therefore faint on the shared scale. Layer 12/head 0 assigns **99.76%** to patches, with a maximum single-patch probability of **10.92%**. These are one-image, two-head examples and do not establish a general rule for layers or heads.

Mean cosine similarity to the layer-12 CLS vector over the four fixed images was 0.004 at layer 1, 0.029 at layer 3, 0.126 at layer 6, 0.175 at layer 9, and 1.000 at layer 12. Mean drift from the prior selected layer was 0.222 (1→3), 0.150 (3→6), 0.550 (6→9), and 0.826 (9→12). The last cosine-to-final value is one by construction. These values reflect raw post-block CLS vectors before the final LayerNorm; they should not be read as classification performance or causal attribution.

The outputs are [attention metadata](../results/analysis/cifar10_seed7/attention_metadata.json), [head summary](../results/analysis/cifar10_seed7/attention_summary.csv), [per-image representation table](../results/analysis/cifar10_seed7/representation_drift.csv), and six labeled PNG overlays. For example, [layer 1/head 0](../results/analysis/cifar10_seed7/attention_cifar_train_00000_l01_h00.png) and [layer 12/head 0](../results/analysis/cifar10_seed7/attention_cifar_train_00000_l12_h00.png) show the range. The overlays are attention diagnostics, not explanations, localization maps, or evidence that a region caused the prediction.
