import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from vit_lab.benchmarking.runner import analytical_macs
from vit_lab.models.vision_transformer import VisionTransformer


def test_analytical_macs_match_profiled_matrix_and_convolution_flops() -> None:
    config = {
        "image_size": 16, "patch_size": 4, "in_channels": 3,
        "embed_dim": 24, "depth": 2, "num_heads": 4,
        "mlp_ratio": 4.0, "num_classes": 3,
    }
    model = VisionTransformer(**config).eval()
    image = torch.randn(1, 3, 16, 16)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as recorded:
        with torch.inference_mode():
            model(image)
    matrix_and_convolution_flops = sum(
        event.flops for event in recorded.key_averages()
        if event.key in ("aten::conv2d", "aten::addmm", "aten::bmm")
    )
    counts = analytical_macs(config, 16)
    assert counts["tokens"] == 17
    assert counts["flops_2_per_mac"] == matrix_and_convolution_flops
    assert counts["all_layer_attention_map_bytes_fp32"] == 2 * 4 * 17**2 * 4


def test_analytical_macs_reject_non_divisible_resolution() -> None:
    with pytest.raises(ValueError, match="divisible"):
        analytical_macs({"patch_size": 4}, 17)


def test_mean_pooling_mac_count_matches_profiled_work() -> None:
    config = {
        "image_size": 16, "patch_size": 4, "in_channels": 3,
        "embed_dim": 24, "depth": 2, "num_heads": 4,
        "mlp_ratio": 4.0, "num_classes": 3, "pooling": "mean",
    }
    model = VisionTransformer(**config).eval()
    image = torch.randn(1, 3, 16, 16)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as recorded:
        with torch.inference_mode():
            model(image)
    matrix_and_convolution_flops = sum(
        event.flops for event in recorded.key_averages()
        if event.key in ("aten::conv2d", "aten::addmm", "aten::bmm")
    )
    counts = analytical_macs(config, 16)
    assert counts["tokens"] == 16
    assert counts["flops_2_per_mac"] == matrix_and_convolution_flops
