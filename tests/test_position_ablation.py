import copy

import pytest
import yaml

from vit_lab.experiments import position


def test_position_variants_keep_training_controls_and_freeze(tmp_path, monkeypatch) -> None:
    base = {
        "model": {"image_size": 32, "patch_size": 8, "embed_dim": 96,
                  "depth": 4, "num_heads": 8, "num_classes": 10},
        "training": {"dataset": "cifar10", "seed": 7, "batch_size": 128,
                     "epochs": 20, "learning_rate": 0.0003},
    }
    original = copy.deepcopy(base)
    for mode, seed in position.CASE_ORDER:
        actual = position.variant_config(base, mode=mode, seed=seed)
        expected = copy.deepcopy(base)
        expected["training"]["seed"] = seed
        if mode == "none":
            expected["model"]["position_embedding"] = "none"
        assert actual == expected
    assert base == original
    for seed in position.SEEDS:
        existing = copy.deepcopy(base)
        existing["training"]["seed"] = seed
        (tmp_path / f"p08seed{seed}.yaml").write_text(yaml.safe_dump(existing))
    monkeypatch.setattr(position, "_existing_config",
                        lambda seed: tmp_path / f"p08seed{seed}.yaml")
    directory = tmp_path / "positions"
    cases = position.prepare_configs(directory)
    assert len(cases) == 6
    assert len(list(directory.glob("*.yaml"))) == 3
    assert position.prepare_configs(directory) == cases
    (directory / "position_none_seed7.yaml").write_text("model: changed\n")
    with pytest.raises(ValueError, match="Frozen position config differs"):
        position.prepare_configs(directory)
