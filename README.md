# Decomposed Velocity Fields During Grokking

[![CI](https://github.com/tomasblood/grokking-velocity-hodge/actions/workflows/ci.yml/badge.svg)](https://github.com/tomasblood/grokking-velocity-hodge/actions/workflows/ci.yml)

This repository contains the submission manuscript and the Grokking
reproduction code derived from
[`tomasblood/geometric-ml-thesis-sample`](https://github.com/tomasblood/geometric-ml-thesis-sample).

The manuscript in `paper.tex` is assembled exclusively from wording that
appears in the final thesis source. LaTeX structure, labels, and file paths were
added to make the material compile as an article.

The repository is deliberately limited to the paper's Grokking case study. It
includes the pointwise velocity-field Hodge decomposition, effective dimension,
diffusion-operator, circular-coordinate, probe-robustness, event-study, and
global Bures--Wasserstein distance analyses. The source thesis repository's
AirXiv, BW-geodesic, spectral-perturbation, and marginal-ellipse analyses are
not included.

## Build the paper

Build `paper.pdf` with:

```powershell
latexmk -pdf paper.tex
```

## Reproduce the experiments

See [`reproducibility/README.md`](reproducibility/README.md) for configuration,
training, analysis, and exact output instructions. The eight-seed Hodge result
can be regenerated immediately from the tracked saved results with:

```powershell
python reproducibility/Grokking/Analysis/12_hodge_cross_seed_summary.py
```

The reusable analysis code is installed from `src/grokking_velocity_hodge`;
the scripts do not depend on notebook-only path injection.

## Current status

The eight-seed empirical Hodge robustness sweep is complete, its saved results
and training metadata are tracked under
`reproducibility/artifacts/eight_seed_hodge`, and the manuscript tables are
generated from those files. The draft remains venue-neutral; submission still
requires the target venue's document class, author metadata, and any venue-specific
data/code availability statement.

## Provenance

The imported reproduction code retains the source repository's MIT licence.
The source repository is configured locally as the `upstream` Git remote.
