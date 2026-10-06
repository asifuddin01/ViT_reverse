# Inside the Vision Transformer

A from-scratch Vision Transformer and reverse-engineering study focused on tensor flow, numerical equivalence, attention diagnostics, and computational cost.

This repository is in active development. The [execution plan](VIT_EXECUTION_PLAN.md) defines the phases and evidence required before results are reported. The [project specification](PROJECT_SPEC.md) preserves the original brief. The pinned reference checkpoint has been inspected and verified; see the [reference contract](docs/reference_contract.md).

## Setup

Use Python 3.12. On macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The reference inventory command downloads the pinned checkpoint into the Hugging Face cache. It verifies its SHA-256 before loading it.

```bash
vit-lab inspect-reference --config configs/vit_base.yaml
vit-lab compare --config configs/vit_base.yaml --offline --device cpu
```

Run tests with `pytest -q`. The [reverse-engineering report](docs/reverse_engineering.md) records the verified CPU/FP32 layer comparison and links its [results](results/tables/equivalence.csv). The [experiment log](docs/experiments.md) records a completed 20-epoch CIFAR-10 CPU baseline: **69.50% validation accuracy** at seed 7, with identical validation predictions after reloading the best checkpoint. The official test split was not used. A fresh training run can be started or an interrupted one resumed with:

```bash
vit-lab train --config configs/vit_tiny_cifar.yaml --run-dir results/runs/cifar_tiny_seed7 --device cpu
vit-lab train --config configs/vit_tiny_cifar.yaml --run-dir results/runs/cifar_tiny_seed7 --device cpu --resume
```

Use the second command only if the first was interrupted after a completed epoch. The [attention analysis](docs/attention_analysis.md) records selected CLS attention maps and representation drift from four fixed real images. Once the official CIFAR-10 training files are in `data/`, reproduce it with:

```bash
vit-lab analyze --config configs/vit_base.yaml --device cpu --offline
```

The current [handoff status](PROJECT_STATUS.md) names the next single task and exact resume steps. Latency, ablation, and retinal-transfer results remain unmeasured.
