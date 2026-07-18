# Grokking reproduction code

This directory contains the exported code used to reproduce the grokking
experiments reported in `paper.tex`.

## Structure

- `Grokking/Training/01_train_reproducible.py` trains the modular-addition model.
- `Grokking/Analysis/01_velocity_hodge.py` computes the pointwise velocity-field Hodge decomposition.
- `Grokking/Analysis/02_effdim_pca.py` computes effective dimension.
- `Grokking/Analysis/03_eigenvalues.py` computes the diffusion-operator eigenspectra.
- `Grokking/Analysis/05_heat_kernel.py` computes heat-kernel BW distances.
- `Grokking/Analysis/07_circular_coords.py` computes circular coordinates.
- `Grokking/Analysis/09_probe_subset_robustness.py` evaluates probe-subset robustness.
- `Grokking/Analysis/10_event_study.py` constructs the seed-aligned event study.
- `Grokking/Pipelines/main_charts.py` runs the paper analysis stack.
- `Grokking/Pipelines/seed_sweep.py` runs the configured robustness seeds.
- `Shared/` contains common runtime and numerical helpers.

## Configuration

The scripts can run as exported Databricks notebooks or as ordinary Python
programs. Configure their input and output locations with:

- `THESIS_WORKSPACE_ROOT`: directory containing this `reproducibility` folder.
- `THESIS_DATA_ROOT`: root for activations, numerical results, and generated figures.
- `THESIS_SHARED_DIR`: absolute path to `reproducibility/Shared`.
- `GROKKING_ACTIVATION_DIR`: optional override for saved activations.

Install dependencies with:

```powershell
pip install -r reproducibility/requirements.txt
```

Train the canonical model when activations are unavailable, then run:

```powershell
$env:THESIS_SHARED_DIR = (Resolve-Path reproducibility/Shared)
python reproducibility/Grokking/Pipelines/main_charts.py
```

The optional seed sweep is configured in
`Grokking/config/seed_sweep.json` and run with:

```powershell
python reproducibility/Grokking/Pipelines/seed_sweep.py
```
