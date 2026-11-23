# Workflow package refactor

The `src/analysis/workflows.py` monolith is now split into one module per command.
The Python CLI (`src/analysis/cli.py`) does not need any changes.

## Install

From the project root:

```bash
unzip -o ~/Downloads/workflows_refactor.zip -d .
bash install_workflows.sh
```

The installer backs up the original file to `src/analysis/workflows.py.pre_refactor.bak`.
Do not delete the backup until you have run your normal analysis commands.
If you use a local configuration different from the latest source in the
conversation, make sure it is saved before you install.

## Layout

- `workflows/__init__.py`: CLI dispatcher, preserving `workflows.run_analysis(args)`
- `workflows/common.py`: output paths, figure saving, CSV/JSON exports, display,
  and shared mean-fragment-length extraction
- `workflows/strain_rate.py`: rate-sweep collection and scaling
- `workflows/convergence.py`: refinement/sensitivity studies
- `workflows/energy.py`: energy views and conservation diagnostics
- `workflows/distribution.py`: mass/length distributions
- `workflows/history.py`: scalar histories
- `workflows/spatial.py`: 1D fracture-position maps

## Important

No `io/`, `calculations/`, `plots/`, or `cli.py` files are replaced. The same
public CLI arguments and output filenames are retained. Only the workflow
orchestration has been reorganised.

The resulting package includes numerical logic carried over from
`history_spatial_patch.zip` (the latest shared version), not any unseen
changes you may have made locally after that version.

## Verify

```bash
python3 -m src.analysis --help
python3 -m src.analysis convergence --help
python3 -m src.analysis energy --help
python3 -m src.analysis distribution --help
python3 -m src.analysis history --help
python3 -m src.analysis spatial --help
```

For a headless quick check, use a completed case from your dataset with
`--save --no-show` and compare the CSVs to those generated before migration.
