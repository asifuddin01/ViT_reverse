import pytest
import torch

from vit_lab.models.patch_embedding import PatchEmbedding


def test_patch_shape_and_count() -> None:
    layer = PatchEmbedding(image_size=224, patch_size=16, in_channels=3, embed_dim=32)
    actual = layer(torch.randn(2, 3, 224, 224))
    assert actual.shape == (2, 196, 32)
    assert layer.num_patches == 196


def test_convolution_matches_explicit_patch_projection_and_gradients() -> None:
    torch.manual_seed(7)
    layer = PatchEmbedding(image_size=8, patch_size=2, in_channels=3, embed_dim=5)
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    torch.testing.assert_close(layer(x), layer.forward_explicit(x))
    conv_gradient = torch.autograd.grad(layer(x).square().sum(), x)[0]
    explicit_gradient = torch.autograd.grad(layer.forward_explicit(x).square().sum(), x)[0]
    torch.testing.assert_close(conv_gradient, explicit_gradient)


def test_hand_worked_two_by_two_patches() -> None:
    layer = PatchEmbedding(image_size=4, patch_size=2, in_channels=1, embed_dim=1)
    with torch.no_grad():
        layer.proj.weight.copy_(torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]))
        layer.proj.bias.zero_()
    image = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    expected = torch.tensor([[[44.0], [64.0], [124.0], [144.0]]])
    torch.testing.assert_close(layer(image), expected)
    torch.testing.assert_close(layer.forward_explicit(image), expected)


@pytest.mark.parametrize("shape", [(2, 3, 8), (2, 1, 8, 8), (2, 3, 7, 8), (2, 3, 4, 4)])
def test_rejects_invalid_input_shape(shape: tuple[int, ...]) -> None:
    layer = PatchEmbedding(image_size=8, patch_size=2, in_channels=3, embed_dim=5)
    with pytest.raises(ValueError):
        layer(torch.zeros(shape))


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        PatchEmbedding(image_size=7, patch_size=2, in_channels=3, embed_dim=5)
