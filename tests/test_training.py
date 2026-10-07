import torch
from torch.utils.data import DataLoader, TensorDataset

from vit_lab.models.vision_transformer import VisionTransformer
from vit_lab.training.data import stratified_indices
from vit_lab.training.trainer import _run_epoch, learning_rate_for_epoch


def test_stratified_split_is_deterministic_and_disjoint() -> None:
    labels = [0] * 10 + [1] * 10 + [2] * 10
    train, validation = stratified_indices(labels, validation_fraction=0.2, seed=7)
    assert len(train) == 24 and len(validation) == 6
    assert not set(train) & set(validation)
    assert sorted(train + validation) == list(range(30))
    assert [labels[i] for i in validation].count(0) == 2
    assert (train, validation) == stratified_indices(labels, validation_fraction=0.2, seed=7)


def test_learning_rate_warmup_and_decay() -> None:
    first = learning_rate_for_epoch(0, base_rate=0.001, epochs=10, warmup_epochs=2)
    second = learning_rate_for_epoch(1, base_rate=0.001, epochs=10, warmup_epochs=2)
    last = learning_rate_for_epoch(9, base_rate=0.001, epochs=10, warmup_epochs=2)
    assert first == 0.0005
    assert second == 0.001
    assert 0 < last < second


def test_one_training_and_validation_step() -> None:
    torch.manual_seed(7)
    model = VisionTransformer(
        image_size=16, patch_size=4, embed_dim=24, depth=1,
        num_heads=4, num_classes=3,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(6, 3, 16, 16), torch.tensor([0, 1, 2, 0, 1, 2])),
        batch_size=3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_loss, train_accuracy, _ = _run_epoch(
        model, loader, device="cpu", optimizer=optimizer, max_batches=1
    )
    val_loss, val_accuracy, logits = _run_epoch(
        model, loader, device="cpu", optimizer=None, max_batches=1
    )
    assert train_loss > 0 and val_loss > 0
    assert 0 <= train_accuracy <= 1 and 0 <= val_accuracy <= 1
    assert logits.shape == (3, 3)



def test_run_epoch_upsamples_native_images_to_model_size() -> None:
    torch.manual_seed(7)
    model = VisionTransformer(image_size=16, patch_size=8, embed_dim=24, depth=1,
                              num_heads=4, num_classes=3)
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 8, 8), torch.tensor([0, 1, 2, 0])),
                        batch_size=2)
    _, _, logits = _run_epoch(model, loader, device="cpu", optimizer=None, max_batches=None)
    upsampled = torch.nn.functional.interpolate(loader.dataset.tensors[0], size=(16, 16),
                                                mode="bicubic", align_corners=False, antialias=True)
    with torch.inference_mode():
        torch.testing.assert_close(logits, model.eval()(upsampled), rtol=0, atol=1e-6)
