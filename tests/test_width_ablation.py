import copy

import pytest
import yaml

from vit_lab.experiments import width


def test_width_variants_keep_training_controls_and_freeze(tmp_path, monkeypatch) -> None:
    base = {
        "model": {"image_size": 32, "patch_size": 8, "embed_dim": 96,
                  "depth": 4, "num_heads": 8, "num_classes": 10},
        "training": {"dataset": "cifar10", "seed": 7, "batch_size": 128,
                     "epochs": 20, "learning_rate": 0.0003},
    }
    original = copy.deepcopy(base)
    for embed_dim, seed in width.CASE_ORDER:
        actual = width.variant_config(base, width=embed_dim, seed=seed)
        expected = copy.deepcopy(base)
        expected["model"]["embed_dim"] = embed_dim
        expected["training"]["seed"] = seed
        assert actual == expected
    assert base == original
    for seed in width.SEEDS:
        existing = copy.deepcopy(base)
        existing["training"]["seed"] = seed
        (tmp_path / f"p08seed{seed}.yaml").write_text(yaml.safe_dump(existing))
    monkeypatch.setattr(width, "_existing_config",
                        lambda seed: tmp_path / f"p08seed{seed}.yaml")
    directory = tmp_path / "width"
    cases = width.prepare_configs(directory)
    assert len(cases) == 6
    assert len(list(directory.glob("*.yaml"))) == 3
    assert width.prepare_configs(directory) == cases
    (directory / "width_d192_seed7.yaml").write_text("model: changed\n")
    with pytest.raises(ValueError, match="Frozen width config differs"):
        width.prepare_configs(directory)
