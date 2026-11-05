# Proposed Cloud Environment Configuration

The onboarding draft was read successfully but rejected an update because it was no longer editable (already published or publication in progress). The tested configuration is preserved here for a future editable setup draft.

## Install script

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /workspace/datamining-project
python -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
.venv/bin/python -m src.download_data --config configs/experiment.yaml
```

## Start instructions

Work in the existing checkout at `/workspace/datamining-project`. Do not create a Git worktree unless the user explicitly requests one.

Read `PROJECT_CONTEXT.md` and `reports/stage_gates.md` before changing research scope. Stages 3–8 are intentionally not implemented.

Use `/workspace/datamining-project/.venv/bin/python`. For plotting commands, set:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR=/workspace/datamining-project/.cache/matplotlib
export XDG_CACHE_HOME=/workspace/datamining-project/.cache
```

Verify the prepared environment with:

```bash
cd /workspace/datamining-project
./scripts/run_stage_0_2.sh
```

The runner verifies pinned public dataset hashes, rebuilds Stage 0–2 artifacts, and runs the tests. Raw and processed datasets are intentionally Git-ignored. No service process needs to start.

No additional secrets or runtime variables are required. The current instance used an enforced unrestricted network policy for the public Kaggle download; no network configuration change was requested.
