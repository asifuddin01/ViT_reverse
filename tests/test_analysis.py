import pytest
import torch
from PIL import Image

from vit_lab.analysis.diagnostics import capture_selected, render_overlay, representation_rows
from vit_lab.models.vision_transformer import VisionTransformer


def test_selective_capture_matches_full_analysis_and_removes_hooks() -> None:
    torch.manual_seed(7)
    model = VisionTransformer(
        image_size=16, patch_size=4, embed_dim=24, depth=2,
        num_heads=4, num_classes=3,
    ).eval()
    image = torch.randn(1, 3, 16, 16)
    with torch.inference_mode():
        full = model(image, return_attention=True, return_hidden_states=True)
    logits, maps, hidden = capture_selected(
        model, image, attention_layers=(1, 2), heads=(0, 2), hidden_layers=(1, 2)
    )
    torch.testing.assert_close(logits, full.logits)
    for layer in (1, 2):
        torch.testing.assert_close(hidden[layer], full.hidden_states[layer - 1][0, 0])
        for head in (0, 2):
            row = maps[layer, head]
            torch.testing.assert_close(row, full.attention_maps[layer - 1][0, head, 0])
            assert row.numel() == 17
            assert row.sum().item() == pytest.approx(1.0, abs=1e-6)
        assert not model.blocks[layer - 1]._forward_hooks
        assert not model.blocks[layer - 1].attn.attn_drop._forward_hooks
    logits_again, maps_again, _ = capture_selected(
        model, image, attention_layers=(1, 2), heads=(0, 2), hidden_layers=()
    )
    assert torch.equal(logits, logits_again)
    assert all(torch.equal(maps[key], maps_again[key]) for key in maps)


def test_representation_drift_and_overlay(tmp_path) -> None:
    hidden = {1: torch.tensor([1.0, 0.0]), 2: torch.tensor([0.0, 1.0])}
    rows = representation_rows("sample", hidden)
    assert rows[0]["cosine_to_final"] == pytest.approx(0.0)
    assert rows[0]["drift_from_previous_selected"] is None
    assert rows[1]["cosine_to_final"] == pytest.approx(1.0)
    assert rows[1]["drift_from_previous_selected"] == pytest.approx(1.0)
    output = tmp_path / "overlay.png"
    render_overlay(
        torch.zeros(3, 16, 16), torch.tensor([0.2, 0.1, 0.2, 0.3, 0.2]),
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), maximum=0.3,
        image_id="sample", layer=1, head=0, model_id="tiny", output=output,
    )
    with Image.open(output) as image:
        assert image.size == (600, 550)
