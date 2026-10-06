# Instructions for anyone continuing this project

1. Read `PROJECT_STATUS.md` first. It records the latest completed task, evidence, next action, and any blockers. Then read `VIT_EXECUTION_PLAN.md` and `PROJECT_SPEC.md` for the full scope.
2. Work on one task at a time. Finish its implementation, run the relevant checks, update `PROJECT_STATUS.md`, and commit before starting the next task.
3. Keep the project in `/Users/mdasifuddin/AI/ViT_reverse` and use the `origin` remote configured for `https://github.com/asifuddin01/ViT_reverse.git`.
4. Use Python 3.12 in `.venv`. Install with `python -m pip install -e '.[dev]'`; install the `experiments` extra only when needed. Run `pytest -q` after model or reference changes.
5. Preserve the pinned reference model/revision/checksum in `configs/vit_base.yaml`. If changing it, document why and rerun the full equivalence report.
6. Record real experiment results with configs, environment, seeds, input splits, and commands. Do not invent accuracy or benchmark numbers. Keep datasets and model checkpoints out of Git; commit small tables and documentation.
7. Before stopping, leave a clean or clearly described Git state. Update the handoff file with the exact next command and any push/authentication issue so another AI can resume without guessing.

User requests in the conversation take precedence over this handoff guidance and the project plan.
