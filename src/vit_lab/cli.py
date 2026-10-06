"""Command-line entry points for reproducible project tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from vit_lab.reference.loader import inspect_reference


def main() -> None:
    parser = argparse.ArgumentParser(prog="vit-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-reference", help="Inventory the pinned timm reference")
    inspect.add_argument("--config", type=Path, default=Path("configs/vit_base.yaml"))
    inspect.add_argument(
        "--output", type=Path, default=Path("results/logs/reference_inventory.json")
    )
    inspect.add_argument("--no-weights", action="store_true", help="Inspect architecture only")
    inspect.add_argument("--offline", action="store_true", help="Use cached Hub files only")
    inspect.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    if args.command == "inspect-reference":
        with args.config.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        inventory = inspect_reference(
            config["reference"],
            load_weights=not args.no_weights,
            local_files_only=args.offline,
            cache_dir=args.cache_dir,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
        print(
            f"{inventory['reference']['architecture']}: "
            f"{inventory['model']['parameter_count']:,} parameters; "
            f"{inventory['feature_shape']} features; {inventory['logit_shape']} logits"
        )


if __name__ == "__main__":
    main()
