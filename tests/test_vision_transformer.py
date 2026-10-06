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


def test_invalid_attention_and_positions() -> None:
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(embed_dim=25, num_heads=4)
    with pytest.raises(ValueError):
        add_position(torch.zeros(1, 5, 8), torch.zeros(1, 4, 8))
    with pytest.raises(ValueError):
        resize_position_embedding(
            torch.zeros(1, 7, 8), old_grid=(2, 2), new_grid=(3, 3)
        )
