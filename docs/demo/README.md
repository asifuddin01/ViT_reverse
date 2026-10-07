# Demo

| File | What it shows | How to regenerate |
|---|---|---|
| [`vit_lab_demo.mp4`](vit_lab_demo.mp4) | 65-second walkthrough: tests, the equivalence check, real-photo predictions, main results, the Colab T4 run | `python scripts/make_demo_video.py` (needs `pip install -e '.[demo]'`) |
| [`demo_predictions.png`](demo_predictions.png), [`.json`](demo_predictions.json) | Top-5 ImageNet classes from the from-scratch ViT-Base with the verified weights, plus last-layer CLS attention | `python scripts/demo.py --offline` |
| [`screenshots/terminal_*.png`](screenshots/) | Keyframes of the replayed terminal sessions | written by `make_demo_video.py` |
| [`screenshots/colab_*.png`](screenshots/) | The Colab Tesla T4 run of `notebooks/colab_t4_followups.ipynb`: overview, CUDA equivalence, CUDA benchmark, a 128-px training log, RetinaMNIST comparison | captured from the finished notebook; the account header is cropped out |
| [`logs/`](logs/) | Raw output of `pytest -q`, `vit-lab compare … --image data/cifar_train_00000.png`, and `scripts/demo.py`, replayed in the video | rerun those commands and redirect their output |

The terminal scenes are not invented: they replay the saved logs as typed commands.

In `demo.py` the timm reference keeps its default fused attention, so its logits differ from the educational model by about 1e-5. The strict equivalence check builds timm with fused attention disabled, and then the difference is exactly 0.0.

The last-layer CLS attention concentrates on a few background patches. This is a known ViT artifact ([Darcet et al., 2023, "Vision Transformers Need Registers"](https://arxiv.org/abs/2309.16588)). It is a diagnostic, not an explanation of the prediction.

**Image credits.** The photos are scikit-image's bundled samples `chelsea` (CC0, Stéfan van der Walt), `coffee` (CC0, Rachel Michetti, courtesy of Pikolo Espresso Bar), and `rocket` (the SpaceX DSCOVR launch photo, released to the public domain). See the scikit-image `data` module documentation.
