"""Network-free oracle test against a small timm VisionTransformer."""

import timm
import torch
from timm.layers import set_fused_attn, use_fused_attn

from vit_lab.models.vision_transformer import VisionTransformer
from vit_lab.reference.compare import _capture
from vit_lab.reference.weight_map import load_exact_reference_weights


def test_small_model_matches_timm_with_identical_weights() -> None:
    prior_fusion = use_fused_attn()
    set_fused_attn(False)
    try:
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
    finally:
        set_fused_attn(prior_fusion)
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
    mapping = load_exact_reference_weights(ours, reference)
    assert len(mapping) == len(reference.state_dict())
    assert all(row["transformation"] == "identity" for row in mapping)
    torch.manual_seed(11)
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(
            ours(images), reference(images), rtol=1e-4, atol=1e-5
        )
    reference_stages = _capture(reference, images)
    our_stages = _capture(ours, images)
    assert reference_stages.keys() == our_stages.keys()
    for stage, expected in reference_stages.items():
        torch.testing.assert_close(our_stages[stage], expected, rtol=1e-4, atol=1e-5)
