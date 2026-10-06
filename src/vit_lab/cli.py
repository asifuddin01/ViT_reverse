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
    compare = commands.add_parser("compare", help="Compare every ViT block to the pinned reference")
    compare.add_argument("--config", type=Path, default=Path("configs/vit_base.yaml"))
    compare.add_argument("--output", type=Path, default=Path("results/tables/equivalence.csv"))
    compare.add_argument(
        "--mapping-output", type=Path, default=Path("results/tables/weight_mapping.csv")
    )
    compare.add_argument("--image", type=Path, help="Optional real image for a fourth comparison")
    compare.add_argument("--offline", action="store_true")
    compare.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    compare.add_argument("--rtol", type=float, default=1e-4)
    compare.add_argument("--atol", type=float, default=1e-5)
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
    elif args.command == "compare":
        from vit_lab.reference.compare import compare_reference

        with args.config.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        rows = compare_reference(
            config,
            output=args.output,
            mapping_output=args.mapping_output,
            image_path=args.image,
            local_files_only=args.offline,
            device=args.device,
            rtol=args.rtol,
            atol=args.atol,
        )
        failed = [row for row in rows if not row["passed"]]
        print(f"Wrote {len(rows)} stage comparisons to {args.output}")
        print(f"Wrote parameter mapping to {args.mapping_output}")
        print(f"Failed comparisons: {len(failed)}")
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
