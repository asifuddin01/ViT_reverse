import pytest
import torch

from vit_lab.models.attention import MultiHeadSelfAttention
from vit_lab.models.positional_embedding import add_position, resize_position_embedding
from vit_lab.models.transformer_block import TransformerBlock
from vit_lab.models.vision_transformer import VisionTransformer, VisionTransformerOutput


def test_attention_probabilities_and_gradients() -> None:
    attention = MultiHeadSelfAttention(embed_dim=24, num_heads=4)
    tokens = torch.randn(2, 5, 24, requires_grad=True)
    output, probabilities = attention(tokens, return_attention=True)
    assert output.shape == tokens.shape
    assert probabilities.shape == (2, 4, 5, 5)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 4, 5))
    output.square().sum().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert attention.qkv.weight.grad is not None
    assert attention.proj.weight.grad is not None


def test_block_preserves_shape() -> None:
    block = TransformerBlock(embed_dim=24, num_heads=4)
    tokens = torch.randn(2, 7, 24)
    output, probabilities = block(tokens, return_attention=True)
    assert output.shape == tokens.shape
    assert probabilities.shape == (2, 4, 7, 7)


def test_model_analysis_outputs_and_gradient_flow() -> None:
    model = VisionTransformer(
        image_size=32, patch_size=4, embed_dim=48, depth=2, num_heads=4,
        num_classes=10,
    )
    images = torch.randn(2, 3, 32, 32)
    details = model(images, return_attention=True, return_hidden_states=True)
    assert isinstance(details, VisionTransformerOutput)
    assert details.logits.shape == (2, 10)
    assert details.patch_embeddings.shape == (2, 64, 48)
    assert details.final_features.shape == (2, 48)
    assert len(details.hidden_states) == 2
    assert len(details.attention_maps) == 2
    assert details.attention_maps[0].shape == (2, 4, 65, 65)
    assert model.forward_features(images).shape == (2, 65, 48)
    details.logits.square().mean().backward()
    for name in (
        "patch_embed.proj.weight", "cls_token", "pos_embed",
        "blocks.0.attn.qkv.weight", "blocks.0.mlp.fc1.weight",
        "blocks.0.norm1.weight", "head.weight",
    ):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(), name


def test_eval_forward_is_stable() -> None:
    model = VisionTransformer(
        image_size=16, patch_size=4, embed_dim=24, depth=1,
        num_heads=4, num_classes=3,
    ).eval()
    image = torch.randn(1, 3, 16, 16)
    with torch.inference_mode():
        torch.testing.assert_close(model(image), model(image))


def test_dynamic_position_resize() -> None:
    model = VisionTransformer(
        image_size=16, patch_size=4, embed_dim=24, depth=1,
        num_heads=4, num_classes=3, dynamic_image_size=True,
    )
    assert model(torch.randn(1, 3, 24, 16)).shape == (1, 3)
    positions = torch.randn(1, 17, 24)
    resized = resize_position_embedding(
        positions, old_grid=(4, 4), new_grid=(6, 4), prefix_tokens=1
    )
    assert resized.shape == (1, 25, 24)
    torch.testing.assert_close(resized[:, 0], positions[:, 0])


def test_no_position_mode_removes_all_token_positions() -> None:
    torch.manual_seed(23)
    learned = VisionTransformer(
        image_size=16, patch_size=8, embed_dim=24, depth=1,
        num_heads=4, num_classes=3,
    ).eval()
    torch.manual_seed(23)
    without_positions = VisionTransformer(
        image_size=16, patch_size=8, embed_dim=24, depth=1,
        num_heads=4, num_classes=3, position_embedding="none",
    ).eval()
    for name, value in without_positions.state_dict().items():
        torch.testing.assert_close(value, learned.state_dict()[name], rtol=0, atol=0)
    weights = {name: value for name, value in learned.state_dict().items()
               if name != "pos_embed"}
    without_positions.load_state_dict(weights, strict=True)
    with torch.no_grad():
        learned.pos_embed.zero_()
    assert "pos_embed" not in without_positions.state_dict()
    assert sum(p.numel() for p in learned.parameters()) - sum(
        p.numel() for p in without_positions.parameters()) == 5 * 24
    image = torch.randn(1, 3, 16, 16)
    swapped_patches = torch.cat((image[:, :, :, 8:], image[:, :, :, :8]), dim=3)
    with torch.inference_mode():
        torch.testing.assert_close(learned(image), without_positions(image))
        torch.testing.assert_close(without_positions(image), without_positions(swapped_patches))


def test_mean_pooling_omits_cls_and_preserves_shared_initial_weights() -> None:
    options = dict(image_size=16, patch_size=8, embed_dim=24, depth=1,
                   num_heads=4, num_classes=3)
    torch.manual_seed(31)
    cls_model = VisionTransformer(**options, pooling="cls").eval()
    torch.manual_seed(31)
    mean_model = VisionTransformer(**options, pooling="mean").eval()
    assert mean_model.cls_token is None
    assert mean_model.pos_embed.shape == (1, 4, 24)
    torch.testing.assert_close(mean_model.pos_embed, cls_model.pos_embed[:, 1:], rtol=0, atol=0)
    for name, value in mean_model.state_dict().items():
        if name != "pos_embed":
            torch.testing.assert_close(value, cls_model.state_dict()[name], rtol=0, atol=0)
    assert sum(p.numel() for p in cls_model.parameters()) - sum(
        p.numel() for p in mean_model.parameters()) == 2 * 24
    image = torch.randn(2, 3, 16, 16)
    details = mean_model(image, return_hidden_states=True)
    assert isinstance(details, VisionTransformerOutput)
    assert mean_model.forward_features(image).shape == (2, 4, 24)
    torch.testing.assert_close(details.final_features,
                               mean_model.forward_features(image).mean(dim=1))
    torch.testing.assert_close(details.logits, mean_model.head(details.final_features))


def test_invalid_attention_and_positions() -> None:
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(embed_dim=25, num_heads=4)
    with pytest.raises(ValueError):
        VisionTransformer(position_embedding="unknown")
    with pytest.raises(ValueError):
        add_position(torch.zeros(1, 5, 8), torch.zeros(1, 4, 8))
    with pytest.raises(ValueError):
        resize_position_embedding(
            torch.zeros(1, 7, 8), old_grid=(2, 2), new_grid=(3, 3)
        )
