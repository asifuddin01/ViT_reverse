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
