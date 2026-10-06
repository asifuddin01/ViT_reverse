"""Auditable parameter mapping from the inspected timm reference."""

from __future__ import annotations

import torch
from torch import nn


def load_exact_reference_weights(
    target: nn.Module, reference: nn.Module
) -> list[dict[str, str]]:
    """Copy equal-named tensors, refusing omissions and shape changes."""
    source_state = reference.state_dict()
    target_state = target.state_dict()
    missing = sorted(set(source_state) - set(target_state))
    unexpected = sorted(set(target_state) - set(source_state))
    if missing or unexpected:
        raise ValueError(f"State-dict keys differ: source-only={missing}, target-only={unexpected}")
    mismatched = [
        key for key in source_state if source_state[key].shape != target_state[key].shape
    ]
    if mismatched:
        raise ValueError(f"State-dict shapes differ: {mismatched}")
    target.load_state_dict(source_state, strict=True)
    return [
        {
            "reference_key": key,
            "target_key": key,
            "shape": str(tuple(tensor.shape)),
            "transformation": "identity",
        }
        for key, tensor in source_state.items()
    ]
