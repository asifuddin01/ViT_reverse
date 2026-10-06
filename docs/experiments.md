# Experiment log

## CIFAR-10 small-model baseline

**Status:** full seed-7 run in progress. Results will be added from saved metrics after completion. A four-batch smoke run has succeeded and confirmed that a saved best-validation checkpoint reproduces the same validation predictions after reload. Smoke accuracy is only a software check and must not be reported as model performance.

The training task uses the official [torchvision CIFAR-10](https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CIFAR10.html) training split (50,000 images). A deterministic stratified 90/10 split with seed 7 creates 45,000 training and 5,000 validation examples; the official test split stays untouched. The exact selected indices are stored in the run directory. Training uses random crop with four-pixel padding, random horizontal flip, tensor conversion, and fixed channel normalization `(x-0.5)/0.5`. Validation uses only tensor conversion and normalization.

The CPU-feasible scratch model in `configs/vit_tiny_cifar.yaml` has 32×32 input, 4×4 patches, embedding width 96, four blocks, eight heads, and ten classes. The protocol is AdamW, learning rate `3e-4`, weight decay `0.05`, batch size 128, two warm-up epochs followed by cosine decay, and 20 epochs. Best model selection uses validation accuracy only. The run saves `config.yaml`, `split_indices.json`, `environment.json`, per-epoch `metrics.csv`, resumable `last.pt`, best-validation `checkpoint.pt`, and `summary.json`. The trainer checks that validation logits match exactly after reloading the best checkpoint.

```bash
vit-lab train --config configs/vit_tiny_cifar.yaml \
  --run-dir results/runs/cifar_tiny_seed7 --device cpu
```

`results/runs/` and `data/` are ignored by Git. Final small summary tables and plots will be committed when measurements are complete. This training model serves implementation and ablation studies; it is separate from the 224-pixel pretrained ViT-Base reference-equivalence model.
