# Decomposed Velocity Fields During Grokking

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
training, and analysis instructions.

## Current status

The draft is venue-neutral. Before submission it still needs the target venue's
document class, author metadata, any required data/code availability statement,
and the completed Hodge robustness sweep identified in the thesis.

## Provenance

The imported reproduction code retains the source repository's MIT licence.
The source repository is configured locally as the `upstream` Git remote.
