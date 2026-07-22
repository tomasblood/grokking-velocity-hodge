# Grokking reproduction code

This directory contains the training, analysis, validation, and orchestration
scripts used by `paper.tex`. Reusable code and the central configuration live in
the installable `grokking_velocity_hodge` package under `src/`.

## Installation

From the repository root, install the analysis package with:

```powershell
python -m pip install -e .
```

Install the model-training dependencies only when retraining:

```powershell
python -m pip install -e ".[training]"
```

For development and linting:

```powershell
python -m pip install -e ".[dev,training]"
```

## Configuration

`ExperimentConfig` in `src/grokking_velocity_hodge/config.py` is the single
source of truth. Analyses use checkpoint epochs recorded in `training.json` and
fall back to the configured epoch schedule only when metadata is unavailable.

Common optional overrides are:

- `THESIS_DATA_ROOT`: data, results, and generated-figure root; defaults to the repository root.
- `GROKKING_ACTIVATION_DIR`: activation directory; defaults below `THESIS_DATA_ROOT/results`.
- `GROKKING_FIGURE_DIR`: generated-figure directory.
- `GROKKING_PCA_DIM`, `GROKKING_KNN`, `GROKKING_K_SPEC`, and `GROKKING_RESOLVENT_EPS`.
- `GROKKING_HODGE_PCA_DIM` and `GROKKING_HODGE_BASIS`.
- `GROKKING_N_EPOCHS`, `GROKKING_SAVE_EVERY`, and the other training overrides documented by the config class.

No `THESIS_SHARED_DIR` or manual `sys.path` configuration is required.

## Main workflow

Train the canonical model when activation snapshots are unavailable:

```powershell
python reproducibility/Grokking/Training/01_train_reproducible.py
```

Run the paper analysis stack:

```powershell
python reproducibility/Grokking/Pipelines/main_charts.py
```

Use `--dry-run` to inspect the planned tasks without requiring activation data.

The portable seed sweep is configured in
`Grokking/config/seed_sweep.json`; its relative paths resolve from the repository
root:

```powershell
python reproducibility/Grokking/Pipelines/seed_sweep.py
```

## Validation

Run all regression tests, including real DiffusionGeometry calibration of exact,
coexact, and harmonic fields:

```powershell
python -m unittest discover -s reproducibility/tests -v
```

Persist the synthetic calibration result with:

```powershell
python reproducibility/Grokking/Validation/01_synthetic_hodge.py
```

Run the empirical Hodge sensitivity sweep over probe subsets, one-at-a-time
PCA/kNN/basis changes, and a correspondence-permutation null with:

```powershell
python reproducibility/Grokking/Analysis/11_hodge_robustness.py
```

The `GROKKING_HODGE_SWEEP_*` environment variables can reduce or expand this
sweep without editing the analysis source.

Historical BW cache provenance and the comparison command are documented in
`reproducibility/PROVENANCE.md`. GitHub Actions runs linting, compilation, tests,
the real Hodge calibration, and a pipeline dry run on every push and pull request.

## Script map

- `Training/01_train_reproducible.py`: modular-addition training and activations.
- `Analysis/01_velocity_hodge.py`: pointwise velocity Hodge decomposition.
- `Analysis/02_effdim_pca.py`: effective dimension and Fourier diagnostics.
- `Analysis/03_eigenvalues.py`: diffusion-operator eigenspectra.
- `Analysis/04_resolvent_bw.py`: consecutive global resolvent BW distances.
- `Analysis/05_heat_kernel.py`: heat-kernel BW distances.
- `Analysis/07_circular_coords.py`: circular-coordinate recovery.
- `Analysis/09_probe_subset_robustness.py`: probe-subset robustness.
- `Analysis/10_event_study.py`: seed-aligned event study.
- `Analysis/11_hodge_robustness.py`: Hodge parameter, subset, and null checks.
- `Validation/`: synthetic calibration and numerical-provenance audits.
