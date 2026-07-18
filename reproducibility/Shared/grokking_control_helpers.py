import gc
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from grokking_summary_helpers import effective_dimension, mean_sd, summarise_transition_series
from runtime import (
    build_normalised_laplacian,
    bw_distance,
    covariance_in_reference_basis,
    heat_kernel_in_reference_basis,
    laplacian_spectrum,
)


def aggregate_probe_subsets(subsets: list[dict], tau_values: list[float]) -> dict:
    eff0 = [s["effective_dimension_summary"]["d_eff_0"] for s in subsets]
    eff25 = [s["effective_dimension_summary"]["d_eff_25000"] for s in subsets]
    eff_ratio = [s["effective_dimension_summary"]["ratio_25000_over_0"] for s in subsets]

    bw_init = [s["bw_resolvent"]["summary"]["initial_0_500"] for s in subsets]
    bw_peak = [s["bw_resolvent"]["summary"]["transition_peak"] for s in subsets]
    bw_post = [s["bw_resolvent"]["summary"]["post_mean"] for s in subsets]
    bw_ratio = [s["bw_resolvent"]["summary"]["transition_peak_over_post_mean"] for s in subsets]

    heat = {}
    for tau in tau_values:
        key = str(tau)
        heat[key] = {
            "transition_peak": mean_sd([s["heat_kernel"]["summary"][key]["transition_peak"] for s in subsets]),
            "post_mean": mean_sd([s["heat_kernel"]["summary"][key]["post_mean"] for s in subsets]),
            "transition_peak_over_post_mean": mean_sd(
                [s["heat_kernel"]["summary"][key]["transition_peak_over_post_mean"] for s in subsets]
            ),
            "n_subsets_transition_peak_exceeds_post_mean": int(
                sum(
                    s["heat_kernel"]["summary"][key]["transition_peak"]
                    > s["heat_kernel"]["summary"][key]["post_mean"]
                    for s in subsets
                )
            ),
        }

    return {
        "effective_dimension": {
            "d_eff_0": mean_sd(eff0),
            "d_eff_25000": mean_sd(eff25),
            "ratio_25000_over_0": mean_sd(eff_ratio),
            "n_subsets_drop": int(sum(a > b for a, b in zip(eff0, eff25))),
        },
        "bw_resolvent": {
            "initial_0_500": mean_sd(bw_init),
            "transition_peak": mean_sd(bw_peak),
            "post_mean": mean_sd(bw_post),
            "transition_peak_over_post_mean": mean_sd(bw_ratio),
            "n_subsets_transition_peak_exceeds_post_mean": int(sum(a > b for a, b in zip(bw_peak, bw_post))),
            "n_subsets_initial_exceeds_transition_peak": int(sum(a > b for a, b in zip(bw_init, bw_peak))),
        },
        "heat_kernel": heat,
    }


# subset checks reuse the canonical activations
def analyse_probe_subset(
    act_dir: str | Path,
    seed: int,
    subset_idx: np.ndarray,
    epochs: list[int],
    tau_values: list[float],
    pca_dim: int = 20,
    knn: int = 15,
    k_spec: int = 30,
    eps: float = 0.01,
) -> dict:
    act_dir = Path(act_dir)
    print(f"subset seed={seed}, n={len(subset_idx)}")
    eff_dims = {}
    covariances = {}
    heat_covs = {tau: {} for tau in tau_values}

    x_ref = np.load(act_dir / "act_0.npy").astype(np.float64)[subset_idx]
    x_ref_red = PCA(n_components=pca_dim, svd_solver="full").fit_transform(x_ref)
    lap_ref = build_normalised_laplacian(x_ref_red, k=knn)
    _, ref_basis = laplacian_spectrum(lap_ref, k=k_spec)
    del x_ref, x_ref_red, lap_ref
    gc.collect()

    valid_epochs = []
    for epoch in epochs:
        path = act_dir / f"act_{epoch}.npy"
        assert path.exists(), f"Missing required activation snapshot: {path}"
        x = np.load(path).astype(np.float64)[subset_idx]
        eff_dims[str(epoch)] = effective_dimension(x)
        x_red = PCA(n_components=pca_dim, svd_solver="full").fit_transform(x)
        lap = build_normalised_laplacian(x_red, k=knn)
        evals, evecs = laplacian_spectrum(lap, k=k_spec)
        covariances[epoch] = covariance_in_reference_basis(evals, evecs, ref_basis, eps=eps)
        for tau in tau_values:
            heat_covs[tau][epoch] = heat_kernel_in_reference_basis(evals, evecs, ref_basis, tau=tau)
        valid_epochs.append(epoch)
        print(f"  epoch {epoch:5d}: d_eff={eff_dims[str(epoch)]:.2f}, lambda1={evals[0]:.4f}")
        del x, x_red, lap, evals, evecs
        gc.collect()

    bw_midpoints = []
    bw_distances = []
    heat_series = {str(tau): {"midpoints": [], "distances": []} for tau in tau_values}
    for epoch_a, epoch_b in zip(valid_epochs[:-1], valid_epochs[1:]):
        midpoint = (epoch_a + epoch_b) / 2.0
        bw_midpoints.append(midpoint)
        bw_distances.append(bw_distance(covariances[epoch_a], covariances[epoch_b]))
        for tau in tau_values:
            heat_series[str(tau)]["midpoints"].append(midpoint)
            heat_series[str(tau)]["distances"].append(bw_distance(heat_covs[tau][epoch_a], heat_covs[tau][epoch_b]))

    eff_summary = {
        "d_eff_0": eff_dims.get("0"),
        "d_eff_3000": eff_dims.get("3000"),
        "d_eff_4000": eff_dims.get("4000"),
        "d_eff_25000": eff_dims.get("25000"),
        "drop_0_to_25000": eff_dims.get("0") - eff_dims.get("25000"),
        "ratio_25000_over_0": eff_dims.get("25000") / eff_dims.get("0"),
    }

    return {
        "seed": seed,
        "subset_size": int(len(subset_idx)),
        "subset_idx": [int(i) for i in subset_idx],
        "valid_epochs": [int(x) for x in valid_epochs],
        "effective_dimensions": eff_dims,
        "effective_dimension_summary": eff_summary,
        "bw_resolvent": {
            "midpoints": [float(x) for x in bw_midpoints],
            "distances": [float(x) for x in bw_distances],
            "summary": summarise_transition_series(bw_midpoints, bw_distances),
        },
        "heat_kernel": {
            "series": heat_series,
            "summary": {
                str(tau): summarise_transition_series(
                    heat_series[str(tau)]["midpoints"],
                    heat_series[str(tau)]["distances"],
                )
                for tau in tau_values
            },
        },
    }
