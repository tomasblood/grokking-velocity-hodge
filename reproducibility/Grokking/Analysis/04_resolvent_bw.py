"""Compute consecutive global resolvent Bures--Wasserstein distances."""

import gc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from grokking_velocity_hodge.config import ExperimentConfig
from grokking_velocity_hodge.runtime import (
    VALIDATION_COLOR,
    build_normalised_laplacian,
    bw_distance,
    configure_grokking_runtime,
    covariance_in_reference_basis,
    laplacian_spectrum,
    load_training_meta,
    set_paper_style,
    shade_grokking_window,
    write_json,
)


def main() -> None:
    GROKKING = configure_grokking_runtime()
    CONFIG = ExperimentConfig.from_environment()
    ACT_DIR = GROKKING.activation_dir
    FIG_DIR = GROKKING.figure_dir
    OUT_DIR = GROKKING.result_dir("grokking_resolvent_bw")
    set_paper_style()

    PCA_DIM = CONFIG.pca_dim
    PCA_SOLVER = CONFIG.pca_solver
    KNN = CONFIG.knn
    K_SPEC = CONFIG.spectral_components
    EPS = CONFIG.resolvent_epsilon

    training = load_training_meta(ACT_DIR)
    EPOCHS = CONFIG.checkpoint_epochs(training)
    val_accs = training.get("val_accs", [])
    meta_epochs = training.get("epochs", training.get("saved_epochs", list(range(len(val_accs)))))

    reference = np.load(ACT_DIR / "act_0.npy").astype(np.float64)
    reference_reduced = PCA(n_components=PCA_DIM, svd_solver=PCA_SOLVER).fit_transform(reference)
    reference_laplacian = build_normalised_laplacian(reference_reduced, k=KNN)
    _, reference_basis = laplacian_spectrum(reference_laplacian, k=K_SPEC)
    del reference, reference_reduced, reference_laplacian

    covariances = {}
    valid_epochs = []
    for epoch in EPOCHS:
        path = ACT_DIR / f"act_{epoch}.npy"
        assert path.exists(), f"Missing required activation snapshot: {path}"
        activation = np.load(path).astype(np.float64)
        reduced = PCA(n_components=PCA_DIM, svd_solver=PCA_SOLVER).fit_transform(activation)
        laplacian = build_normalised_laplacian(reduced, k=KNN)
        eigenvalues, eigenvectors = laplacian_spectrum(laplacian, k=K_SPEC)
        covariances[epoch] = covariance_in_reference_basis(
            eigenvalues,
            eigenvectors,
            reference_basis,
            eps=EPS,
        )
        valid_epochs.append(epoch)
        print(f"epoch {epoch:5d}: lambda_1={eigenvalues[0]:.4f}")
        del activation, reduced, laplacian, eigenvalues, eigenvectors
        gc.collect()

    midpoints = []
    distances = []
    for epoch_a, epoch_b in zip(valid_epochs[:-1], valid_epochs[1:]):
        distance = bw_distance(covariances[epoch_a], covariances[epoch_b])
        midpoints.append((epoch_a + epoch_b) / 2.0)
        distances.append(distance)
        print(f"d_BW({epoch_a}, {epoch_b}) = {distance:.4f}")

    payload = {
        "config": {
            "pca_dim": PCA_DIM,
            "pca_solver": PCA_SOLVER,
            "knn": KNN,
            "k_spec": K_SPEC,
            "eps": EPS,
        },
        "valid_epochs": valid_epochs,
        "bw_distances_consecutive": {
            "midpoint_epochs": [float(epoch) for epoch in midpoints],
            "distances": [float(distance) for distance in distances],
        },
    }
    output_path = write_json(OUT_DIR / "resolvent_bw_results.json", payload)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(midpoints, distances, color="#222222", marker="o", markersize=3, linewidth=1.8)
    shade_grokking_window(axes[0], label=True)
    axes[0].set_ylabel("Consecutive $d_{BW}$")
    axes[0].set_title("Resolvent BW distance between consecutive checkpoints")
    axes[0].legend(loc="upper right")

    axes[1].plot(meta_epochs, val_accs, color=VALIDATION_COLOR, linewidth=1.8)
    shade_grokking_window(axes[1])
    axes[1].set_xlabel("Training epoch")
    axes[1].set_ylabel("Val acc")
    axes[1].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_resolvent_bw.pdf")
    fig.savefig(FIG_DIR / "fig_resolvent_bw.png")
    plt.close(fig)

    assert len(distances) == len(valid_epochs) - 1
    assert output_path.exists()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
