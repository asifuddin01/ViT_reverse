import copy

import pytest
import yaml

from vit_lab.experiments import pooling


def test_pooling_variants_keep_training_controls_and_freeze(tmp_path, monkeypatch) -> None:
    base = {
        "model": {"image_size": 32, "patch_size": 8, "embed_dim": 96,
                  "depth": 4, "num_heads": 8, "num_classes": 10},
        "training": {"dataset": "cifar10", "seed": 7, "batch_size": 128,
                     "epochs": 20, "learning_rate": 0.0003},
    }
    original = copy.deepcopy(base)
    for mode, seed in pooling.CASE_ORDER:
        actual = pooling.variant_config(base, mode=mode, seed=seed)
        expected = copy.deepcopy(base)
        expected["training"]["seed"] = seed
        if mode == "mean":
            expected["model"]["pooling"] = "mean"
        assert actual == expected
    assert base == original
    for seed in pooling.SEEDS:
        existing = copy.deepcopy(base)
        existing["training"]["seed"] = seed
        (tmp_path / f"p08seed{seed}.yaml").write_text(yaml.safe_dump(existing))
    monkeypatch.setattr(pooling, "_existing_config",
                        lambda seed: tmp_path / f"p08seed{seed}.yaml")
    directory = tmp_path / "pooling"
    cases = pooling.prepare_configs(directory)
    assert len(cases) == 6
    assert len(list(directory.glob("*.yaml"))) == 3
    assert pooling.prepare_configs(directory) == cases
    (directory / "pooling_mean_seed7.yaml").write_text("model: changed\n")
    with pytest.raises(ValueError, match="Frozen pooling config differs"):
        pooling.prepare_configs(directory)
