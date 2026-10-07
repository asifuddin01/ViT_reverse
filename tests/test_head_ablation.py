import copy

import pytest
import yaml

from vit_lab.experiments import head_count


def test_head_variants_keep_other_training_controls_and_freeze(tmp_path, monkeypatch) -> None:
    base = {
        "model": {"image_size": 32, "patch_size": 8, "embed_dim": 96,
                  "depth": 4, "num_heads": 8, "num_classes": 10},
        "training": {"dataset": "cifar10", "seed": 7, "batch_size": 128,
                     "epochs": 20, "learning_rate": 0.0003},
    }
    original = copy.deepcopy(base)
    for heads, seed in head_count.CASE_ORDER:
        actual = head_count.variant_config(base, heads=heads, seed=seed)
        expected = copy.deepcopy(base)
        expected["model"]["num_heads"] = heads
        expected["training"]["seed"] = seed
        assert actual == expected
    assert base == original
    for seed in head_count.SEEDS:
        existing = copy.deepcopy(base)
        existing["training"]["seed"] = seed
        (tmp_path / f"p08seed{seed}.yaml").write_text(yaml.safe_dump(existing))
    monkeypatch.setattr(head_count, "_existing_config",
                        lambda seed: tmp_path / f"p08seed{seed}.yaml")
    directory = tmp_path / "heads"
    cases = head_count.prepare_configs(directory)
    assert len(cases) == 12
    assert len(list(directory.glob("*.yaml"))) == 9
    assert head_count.prepare_configs(directory) == cases
    (directory / "head_h04_seed7.yaml").write_text("model: changed\n")
    with pytest.raises(ValueError, match="Frozen head-count config differs"):
        head_count.prepare_configs(directory)
