"""Demo: classify real photos with the from-scratch ViT running the verified pretrained weights.

For each bundled scikit-image sample photo (CC0/public domain; no download), print the
top-5 ImageNet classes from the educational model, confirm its logits against timm, and
save a figure with the last-layer CLS attention diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from skimage import data
from timm.data import ImageNetInfo

from vit_lab.models.vision_transformer import VisionTransformer
from vit_lab.reference.loader import load_reference_model
from vit_lab.reference.weight_map import load_exact_reference_weights

SAMPLES = {"chelsea (cat)": data.chelsea, "coffee": data.coffee, "rocket launch": data.rocket}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/vit_base.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/demo"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference, hub_config, checksum = load_reference_model(
        config["reference"], local_files_only=args.offline)
    model = VisionTransformer(**config["model"]).eval()
    mapping = load_exact_reference_weights(model, reference)
    print(f"Loaded {config['reference']['repo_id']} @ {config['reference']['revision'][:7]}")
    print(f"SHA-256 {checksum[:16]}... verified; {len(mapping)} tensors mapped with strict=True")
    transform = timm.data.create_transform(
        **timm.data.resolve_data_config(hub_config["pretrained_cfg"]), is_training=False)
    labels = ImageNetInfo()
    grid = model.patch_embed.grid_size
    results = []
    fig, axes = plt.subplots(2, len(SAMPLES), figsize=(4.2 * len(SAMPLES), 8.6))
    for column, (name, load) in enumerate(SAMPLES.items()):
        image = Image.fromarray(load()).convert("RGB")
        inputs = transform(image).unsqueeze(0)
        with torch.inference_mode():
            details = model(inputs, return_attention=True)
            expected = reference(inputs)
        difference = (details.logits - expected).abs().max().item()
        probabilities = F.softmax(details.logits[0], dim=0)
        top = torch.topk(probabilities, 5)
        predictions = [(labels.index_to_description(int(i)).split(",")[0], float(p))
                       for p, i in zip(top.values, top.indices)]
        print(f"\n{name}: max |logit difference| vs timm (fused attention) = {difference:.1e}")
        for label, probability in predictions:
            print(f"  {probability:6.1%}  {label}")
        # Last block, CLS query, mean over heads, patch keys only: a diagnostic, not an explanation.
        attention = details.attention_maps[-1][0, :, 0, 1:].mean(0).reshape(1, 1, grid, grid)
        attention = F.interpolate(attention, size=(224, 224), mode="bilinear", align_corners=False)
        shown = (inputs[0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
        axes[0, column].imshow(shown)
        axes[0, column].set_title(f"{name}\n" + "\n".join(
            f"{p:.1%} {label}" for label, p in predictions[:3]), fontsize=9, loc="left")
        axes[1, column].imshow(shown)
        axes[1, column].imshow(attention[0, 0].numpy(), cmap="inferno", alpha=0.55)
        axes[1, column].set_title("layer 12 CLS attention (mean of 12 heads)", fontsize=9)
        for ax in axes[:, column]:
            ax.axis("off")
        results.append({"image": name, "top5": predictions,
                        "max_abs_logit_difference_vs_timm": difference})
    fig.suptitle("From-scratch ViT-Base/16 with verified pretrained weights "
                 "(attention maps are diagnostics, not explanations)", fontsize=11)
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "demo_predictions.png", dpi=120)
    (args.output_dir / "demo_predictions.json").write_text(json.dumps(
        {"reference": config["reference"], "checkpoint_sha256": checksum,
         "device": "cpu", "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {args.output_dir / 'demo_predictions.png'}")


if __name__ == "__main__":
    main()
