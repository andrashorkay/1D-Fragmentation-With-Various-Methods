# Fragmentation analysis tests

These tests are designed for the refactored `src/analysis/workflows/` package.
They use synthetic DataFrames and temporary directories; no C++ simulation
executables, HPC access or large real datasets are required.

## Install and run

Extract this ZIP **at the project root**, alongside `src/`:

```bash
cd ~/Documents/MA3/project
unzip -o ~/Downloads/fragmentation_analysis_tests.zip -d .
python3 -m pip install pytest
MPLBACKEND=Agg python3 -m pytest tests/analysis -q
```

You need the existing project dependencies (`numpy`, `pandas`, `matplotlib`)
and the installed analysis package. Tests are written for Python 3.10+.

## Coverage

- `test_readers.py`: actual method reader contracts, final saved snapshots,
  common recoverable-energy aliases. The low-level whitespace reader is
  monkeypatched to isolate method-specific behaviour.
- `test_fragmentation.py`: fragment-size/mass statistics; strain-rate power-law.
- `test_statistics.py`: normalised distributions; empirical CDF/CCDF; densities.
- `test_theory.py`: Grady fragment length formula and strain-rate scaling.
- `test_convergence.py`: PD timestep regression values; reference selection;
  relative and absolute changes.
- `test_energy.py`: CZM reversible cohesive energy, PD recorded-total balance,
  zero-energy partition, initial energy and invalid inputs.
- `test_spatial.py`: fracture positions on reference bar, stacked snapshots and
  invalid fragment geometry.
- `test_plots.py`: headless rendering of all major plot families.
- `test_workflows.py`: parser/dispatcher checks and `--output` exports for
  all six commands using synthetic cases.

## Limitations

The test suite is **not** validation of the physical models. It does not
measure crack-initiation accuracy, energy convergence or numerical correctness
of the C++ solvers. Real-case regression tests should be added when stable
reference outputs are available. It also cannot guarantee that every file
format written by a modified C++ solver is parsed correctly: reader tests
isolate the low-level whitespace parser.

If you still use the old monolithic `src/analysis/workflows.py`, first
install the refactored workflows package. The suite expects the command modules
at `src/analysis/workflows/{energy,convergence,...}.py`.
