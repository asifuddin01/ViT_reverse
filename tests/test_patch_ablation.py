import copy

import pytest
import yaml

from vit_lab.experiments.patch_size import CASE_ORDER, prepare_configs, variant_config


def test_patch_variant_changes_only_patch_size_and_paired_seed(tmp_path) -> None:
    base = {
        "model": {"image_size": 32, "patch_size": 4, "embed_dim": 96, "depth": 4,
                  "num_heads": 8, "num_classes": 10},
        "training": {"dataset": "cifar10", "seed": 7, "batch_size": 128,
                     "epochs": 20, "learning_rate": 0.0003},
    }
    untouched = copy.deepcopy(base)
    for patch_size, seed in CASE_ORDER:
        actual = variant_config(base, patch_size=patch_size, seed=seed)
        expected = copy.deepcopy(base)
        expected["model"]["patch_size"] = patch_size
        expected["training"]["seed"] = seed
        assert actual == expected
    assert base == untouched
    base_path = tmp_path / "baseline.yaml"
    base_path.write_text(yaml.safe_dump(base))
    config_dir = tmp_path / "variants"
    first = prepare_configs(base_path, config_dir)
    assert len(first) == 9
    assert prepare_configs(base_path, config_dir) == first
    corrupted = config_dir / "patch_p08_seed7.yaml"
    corrupted.write_text("model: changed\n")
    with pytest.raises(ValueError, match="Frozen variant config differs"):
        prepare_configs(base_path, config_dir)
