# Grokking reproduction code

This directory contains the training, analysis, validation, and orchestration
scripts used by `paper.tex`. Reusable code and the central configuration live in
the installable `grokking_velocity_hodge` package under `src/`.

## Installation

From the repository root, install the analysis and test dependencies with:

```powershell
python -m pip install -e ".[dev]"
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

### Regenerate the reported eight-seed result

The derived empirical records needed for the paper are tracked, so this command
does not require the large activation tensors or model checkpoints:

```powershell
python reproducibility/Grokking/Analysis/12_hodge_cross_seed_summary.py
```

It validates the bundled files against the saved SHA-256 manifest and validates
every record, pairs the actual and permutation-null fields, treats
the eight training seeds as the independent units, computes both fixed and
event-aligned phase summaries, and writes:

- `reproducibility/artifacts/eight_seed_hodge/generated/cross_seed_hodge_summary.json`
- `reproducibility/artifacts/eight_seed_hodge/generated/seed_setting_contrasts.csv`
- `reproducibility/artifacts/eight_seed_hodge/generated/baseline_seed_phase_means.csv`
- `reproducibility/artifacts/eight_seed_hodge/generated/generated_manifest.json`
- `generated/eight_seed_hodge_macros.tex` and the two generated table-row files consumed by `paper.tex`

The saved inputs comprise eight `hodge_robustness_seed*.json` files, eight
`training_seed*.json` files, the remote artifact manifest, aggregate source
summaries, and the synthetic Hodge calibration. The raw activation tensors and
model checkpoints are not tracked because of their size; they can be recreated
by the full workflow below. The derived records are sufficient to audit every
number in the manuscript's eight-seed Hodge tables and intervals.

### Recompute from training

Train the canonical model when activation snapshots are unavailable:

```powershell
python reproducibility/Grokking/Training/01_train_reproducible.py
```

Run the paper analysis stack:

```powershell
python reproducibility/Grokking/Pipelines/main_charts.py
```

Use `--dry-run` to inspect the planned tasks without requiring activation data.

The portable eight-run sweep is configured in
`reproducibility/Grokking/config/seed_sweep.json`; its relative paths resolve from the repository
root. It trains any missing runs, executes the Hodge robustness task through the
main pipeline, and regenerates the cross-seed outputs:

```powershell
python reproducibility/Grokking/Pipelines/seed_sweep.py
```

For fresh outputs stored at the configured run paths, the reporting-only step is:

```powershell
python reproducibility/Grokking/Analysis/12_hodge_cross_seed_summary.py `
  --config reproducibility/Grokking/config/seed_sweep.json
```

The full run creates 51 activation snapshots per seed and is compute- and
storage-intensive. Use `--dry-run` first to inspect all eight runs and tasks.

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

## Software versions

- Python 3.11 or newer; continuous integration uses Python 3.12.
- TransformerLens 2.17.0 for the saved training runs.
- `numpy>=2.1,<2.6`, `scipy>=1.15,<1.18`, `matplotlib>=3.9,<3.11`, and
  `scikit-learn>=1.6,<1.9`.
- DiffusionGeometry commit `f5dc795557d07b32795c0bb6bedf465246d999eb`.
- A TeX distribution with `latexmk` for rebuilding `paper.pdf`.

The authoritative dependency constraints are in `pyproject.toml`; the exact
training-library version is also recorded in every saved `training_seed*.json`.

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
- `Analysis/12_hodge_cross_seed_summary.py`: eight-seed inference, phase sensitivity, and generated paper tables.
- `Validation/`: synthetic calibration and numerical-provenance audits.
