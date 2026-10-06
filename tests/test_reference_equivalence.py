"""Network-free oracle test against a small timm VisionTransformer."""

import timm
import torch

from vit_lab.models.vision_transformer import VisionTransformer


def test_small_model_matches_timm_with_identical_weights() -> None:
    reference = timm.models.vision_transformer.VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=48,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
    ).eval()
    ours = VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=48,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
    ).eval()
    ours.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(11)
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(
            ours(images), reference(images), rtol=1e-4, atol=1e-5
        )
