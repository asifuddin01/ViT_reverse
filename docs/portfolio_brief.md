# Portfolio brief: Inside the Vision Transformer

**For the Claude session editing my portfolio website.** Add this project to my portfolio using the facts and assets below. Every number here comes from a saved, reproducible run in the repository; do not round them differently, add new metrics, or describe results more strongly than written here.

## Task

1. **Inspect the portfolio site first.** Find its framework, how existing projects are listed (data file, MDX/Markdown collection, JSON, or components), and the card/detail-page conventions. Follow them exactly; reuse existing components and styles instead of creating new ones.
2. **Add a project card** for the projects list or grid, using the short description and tags below.
3. **Add a project detail or case-study page**, if the site has them, using the sections below. If the site only has cards, use the card copy plus links.
4. **Copy the assets** listed below into the site's own static/assets folder. Do not hotlink `raw.githubusercontent.com`: it does not serve video with a playable content type. Embed the video with a native `<video controls muted playsinline poster=...>` element.
5. Give every image meaningful alt text (suggestions below), and check that the layout works at phone width.
6. Run the site's build, lint, or type check, and fix anything this change breaks. Then show me the diff before committing.

## Facts

| Field | Value |
|---|---|
| Title | Inside the Vision Transformer |
| Subtitle | A from-scratch PyTorch ViT, verified layer by layer against a pretrained reference |
| Repository | https://github.com/asifuddin01/ViT_reverse |
| Year / status | 2026, complete |
| Role | Solo project: design, implementation, experiments, write-up |
| License | MIT (code). The checkpoint and datasets keep their own terms. |
| Hardware | Apple M1 CPU (8 GB) and a Google Colab Tesla T4 GPU |
| Tags | PyTorch, Vision Transformer, Reverse engineering, Deep learning, Computer vision, timm, Benchmarking, Ablation studies, Transfer learning, Medical imaging (research), Google Colab |

## One-liner (≤ 140 characters)

> Built a Vision Transformer from scratch in PyTorch and proved it matches a pretrained ViT-Base exactly, layer by layer, on CPU and GPU.

## Card description (≈ 50 words)

> A from-scratch PyTorch Vision Transformer that loads all 152 tensors of a pinned pretrained ViT-Base and reproduces every intermediate stage exactly (max difference 0.0, CPU and T4 GPU). It then powers attention diagnostics, CPU/GPU benchmarks, controlled CIFAR-10 ablations over three paired seeds, and a retinal-image transfer study.

## Case-study sections

**Problem.** Vision Transformers are usually used through libraries, so their internals stay a black box. I wanted to rebuild one from basic layers and *prove* it is the same function as a production pretrained model. A matching top-1 class was not enough; I wanted equality at every layer. Then I used it to measure which design choices matter.

**What I built.**
- An inspectable ViT: patch projection, learned positions, a CLS token, packed QKV with hand-written `softmax(QKᵀ/√d)V`, and pre-norm blocks. An optional analysis mode returns the attention maps and hidden states.
- A strict weight mapping from the pinned `timm/vit_base_patch16_224.augreg_in21k_ft_in1k` checkpoint (revision and SHA-256 verified) that refuses any missing key or shape mismatch.
- A layer-by-layer comparison tool, a resumable training pipeline with checkpoint-reload proofs, benchmark tooling, and frozen-protocol ablation runners.
- A Colab T4 notebook for the GPU studies, plus a demo script and video.

**Key results** (copy the numbers exactly):

| Result | Value |
|---|---|
| Pretrained tensors mapped | 152 / 152 (86,567,656 parameters) |
| Layer-by-layer equivalence | 100 / 100 stages, max absolute difference **0.0**, on CPU and on a Tesla T4 |
| Inference, 224 px, batch 1 | 81.0 ms (M1 CPU) · 15.0 ms (T4), vs 14.6 ms for timm's fused attention |
| CIFAR-10, patch size (width 96 / width 192) | P=4: 71.6% / **75.3%** · P=8: 63.4% / 68.7% · P=16: 55.9% / 59.3% |
| Learned vs no position embedding | 63.4% vs 56.0%: learned positions won in every seed |
| Mean-patch vs CLS pooling | 65.2% vs 63.4%: mean pooling won in every seed |
| Attention heads 4 / 8 / 12 / 16 at fixed width | 63.8 / 63.4 / 63.0 / 62.9%: no clear difference |
| RetinaMNIST 5-grade transfer, test QWK | linear probe 0.750 · partial fine-tune 0.775 ± 0.038 · full fine-tune 0.773 ± 0.038 |

CIFAR-10 values are best validation accuracy, given as the mean of three paired seeds; the test split was never used. RetinaMNIST values are on the official 400-image test set.

**Engineering rigor (worth highlighting).**
- Protocols and configs were frozen and committed before each experiment. Each comparison uses three paired seeds that share identical train/validation splits.
- Every trained checkpoint was reloaded and shown to reproduce its saved validation predictions.
- The RetinaMNIST models were selected on validation data only. Their test metrics are recomputed from saved predictions, with bootstrap 95% confidence intervals.
- Negative results are reported as found. Neither fine-tuning method beat the linear probe reliably on QWK, and the head count made no clear difference.
- I found and fixed two GPU-only bugs, a device mismatch in the reload check and a CPU data-loading bottleneck, before trusting the GPU numbers.
- A clean clone installs and passes all 34 tests.

**What I learned.** Matching a pretrained model exactly comes down to small details: LayerNorm epsilon, pre-norm order, QKV split order, and turning off fused attention when you need to compare kernels. Patch size and position information mattered far more than head count in this small-data regime. Last-layer CLS attention concentrates on a few background patches, a known ViT artifact, which is why I present attention maps as diagnostics, not explanations.

**Stack.** Python 3.12, PyTorch, timm, torchvision, scikit-learn, MedMNIST, matplotlib, Google Colab (T4), pytest.

## Assets (copy from the repository)

All paths are relative to https://github.com/asifuddin01/ViT_reverse (branch `main`).

| Use | File | Suggested alt text |
|---|---|---|
| Hero/demo video (65 s, 1.1 MB, 1280×720 MP4) | `docs/demo/vit_lab_demo.mp4` | (video) Walkthrough: tests, layer-by-layer check, real-photo predictions, results, Colab GPU run |
| Video poster / card thumbnail | `docs/demo/screenshots/terminal_demo.png` | Terminal: the from-scratch ViT classifying a cat, an espresso, and a rocket launch |
| Prediction + attention figure | `docs/demo/demo_predictions.png` | Three photos with top-3 predictions and last-layer CLS attention overlays |
| Equivalence proof | `docs/demo/screenshots/terminal_equivalence.png` | Terminal: 100 stage comparisons, 0 failures |
| GPU equivalence | `docs/demo/screenshots/colab_02_cuda_equivalence.png` | Colab T4 output: 100/100 stages within tolerance, max difference 0 |
| GPU benchmark | `docs/demo/screenshots/colab_03_cuda_benchmark.png` | Colab T4 latency and memory for 224/384/512-pixel inputs |
| Patch-size chart | `results/figures/patch_size_ablation.png` | Validation accuracy falls as patch size grows from 4 to 16, three paired seeds |
| Width-192 patch chart | `results/colab_t4/width192_patch/width192_patch_accuracy.png` | Width-192 model: 75.3%, 68.7%, 59.3% for patch sizes 4, 8, 16 |
| Position chart | `results/figures/position_ablation.png` | Learned position embeddings beat none in all three seeds |
| Pooling chart | `results/figures/pooling_ablation.png` | Mean patch pooling beats CLS pooling in all three seeds |
| Transfer chart | `results/colab_t4/retina_full/retina_full_transfer.png` | RetinaMNIST test QWK and balanced accuracy with 95% intervals |

Pick 1 hero (the video, with the terminal poster) plus 3–5 supporting images. Do not use all of them.

## Links to show

- Code: https://github.com/asifuddin01/ViT_reverse
- Full report: https://github.com/asifuddin01/ViT_reverse#readme
- Experiment log: https://github.com/asifuddin01/ViT_reverse/blob/main/docs/experiments.md
- Colab notebook: https://colab.research.google.com/github/asifuddin01/ViT_reverse/blob/main/notebooks/colab_t4_followups.ipynb

## Wording rules (please follow)

- Say "matches the pretrained model exactly (max difference 0.0)". Do not say "outperforms timm" or "faster than timm": the speeds are within about 3% of each other.
- RetinaMNIST is a **research and education exercise, not a clinical tool**. Do not use words like "diagnoses", "detects disease", or "clinical-grade".
- Call attention maps **diagnostics**, not explanations of the model's decisions.
- CIFAR-10 figures are **validation** accuracy from small models trained from scratch for 20 epochs. Do not compare them to state-of-the-art test accuracy.
- Do not invent metrics, users, stars, or impact numbers. If the layout needs a stat badge, use: "152/152 tensors mapped", "0.0 max difference", "34 tests", "3 paired seeds".
- Image credits for the demo photos, if the site shows credits: scikit-image sample images, CC0/public domain.
