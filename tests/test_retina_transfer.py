import copy

import numpy as np
import torch

from vit_lab.experiments.retina import TransferHead, cache_features, classification_metrics
from vit_lab.models.vision_transformer import VisionTransformer


def test_cached_tokens_plus_transfer_head_reproduce_full_model() -> None:
    torch.manual_seed(0)
    model = VisionTransformer(image_size=32, patch_size=8, embed_dim=48, depth=3,
                              num_heads=4, num_classes=5).eval()
    images = torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8)
    data_cfg = {"normalization_mean": [0.5] * 3, "normalization_std": [0.5] * 3}
    tokens, features = cache_features(model, images, frozen_blocks=2, data_cfg=data_cfg,
                                      batch_size=3)
    head = TransferHead(copy.deepcopy(model.blocks[2:]), copy.deepcopy(model.norm),
                        copy.deepcopy(model.head)).eval()
    with torch.inference_mode():
        expected = model((images.permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5)
        torch.testing.assert_close(head(tokens), expected, rtol=0, atol=0)
        torch.testing.assert_close(model.head(features), expected, rtol=0, atol=0)


def test_metrics_on_known_predictions() -> None:
    labels = np.array([0, 1, 2, 3, 4, 0])
    metrics = classification_metrics(labels, labels.copy(), 5)
    assert metrics["quadratic_weighted_kappa"] == 1.0 and metrics["balanced_accuracy"] == 1.0
    constant = classification_metrics(labels, np.zeros_like(labels), 5)
    assert constant["quadratic_weighted_kappa"] == 0.0 and constant["balanced_accuracy"] == 0.2


def test_full_finetune_runs_and_selects_on_validation() -> None:
    from vit_lab.experiments.retina_full import finetune_full

    torch.manual_seed(0)
    backbone = VisionTransformer(image_size=32, patch_size=8, embed_dim=48, depth=2,
                                 num_heads=4, num_classes=10).eval()
    images = {s: torch.randint(0, 256, (6, 32, 32, 3), dtype=torch.uint8)
              for s in ("train", "val", "test")}
    labels = {s: np.array([0, 1, 2, 3, 4, 0]) for s in images}
    cfg = {"batch_size": 4, "epochs": 2, "learning_rate": 1e-3, "weight_decay": 0.0,
           "warmup_epochs": 1, "mixed_precision": True,
           "augmentation": {"random_resized_crop_scale": [0.8, 1.0], "horizontal_flip": True}}
    data_cfg = {"size": 32, "normalization_mean": [0.5] * 3, "normalization_std": [0.5] * 3}
    result = finetune_full(backbone, images, labels, cfg, data_cfg, seed=7, num_classes=5,
                           device="cpu")
    assert len(result["history"]) == 2 and result["best_epoch"] in (1, 2)
    assert result["rows"]["test"][1].shape == (6, 5) and not result["mixed_precision"]
    assert backbone.head.out_features == 10  # The verified backbone is not modified.
