"""Track diffusion-operator eigenspectra across training."""

import gc

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grokking_velocity_hodge.config import ExperimentConfig
from grokking_velocity_hodge.runtime import (
    VALIDATION_COLOR,
    build_normalised_laplacian,
    configure_grokking_runtime,
    laplacian_spectrum,
    load_training_meta,
    set_paper_style,
    shade_grokking_window,
)


def main() -> None:
    GROKKING = configure_grokking_runtime()
    CONFIG = ExperimentConfig.from_environment()
    ROOT = GROKKING.root
    ACT_DIR = GROKKING.activation_dir
    FIG_DIR = GROKKING.figure_dir
    set_paper_style()
    print(f"ROOT: {ROOT}")

    PCA_DIM = CONFIG.pca_dim
    PCA_SOLVER = CONFIG.pca_solver
    KNN = CONFIG.knn
    EPS = CONFIG.resolvent_epsilon

    training = load_training_meta(ACT_DIR)
    EPOCHS = CONFIG.checkpoint_epochs(training)
    train_accs = training.get("train_accs", [])
    val_accs = training.get("val_accs", [])
    meta_epochs = training.get("saved_epochs", training.get("epochs", []))
    print(
        f"Training metadata: {len(meta_epochs)} epochs, "
        f"final val acc = {val_accs[-1]:.3f}"
        + (f", final train acc = {train_accs[-1]:.3f}" if train_accs else "")
    )

    K_SHOW = 10

    all_evals = []
    valid_epochs = []

    for ep in EPOCHS:
        path = ACT_DIR / f"act_{ep}.npy"
        assert path.exists(), f"Missing required activation snapshot: {path}"
        X = np.load(path).astype(np.float64)
        pca = PCA(n_components=PCA_DIM, svd_solver=PCA_SOLVER)
        X_red = pca.fit_transform(X)

        L = build_normalised_laplacian(X_red, k=KNN)
        evals, _ = laplacian_spectrum(L, k=K_SHOW)

        all_evals.append(evals)
        valid_epochs.append(ep)
        print(f"  epoch {ep:>6d}: lambda_1={evals[0]:.4f}, lambda_{K_SHOW}={evals[-1]:.4f}")
        gc.collect()

    all_evals = np.array(all_evals)  # (n_epochs, K_SHOW)
    print(f"\nCollected spectra for {len(valid_epochs)} epochs")

    fig, axes = plt.subplots(
        3, 1, figsize=(8.8, 9.2), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.5]}
    )

    # --- Panel 1: Laplacian eigenvalues
    ax = axes[0]
    cmap = plt.cm.viridis
    for i in range(K_SHOW):
        color = cmap(i / K_SHOW)
        ax.plot(
            valid_epochs,
            all_evals[:, i],
            "-o",
            markersize=2.5,
            color=color,
            label=f"$\\lambda_{{{i + 1}}}$",
            linewidth=1.2,
        )
    shade_grokking_window(ax, label=True)
    ax.set_ylabel("Laplacian eigenvalue $\\lambda_k$")
    ax.set_title("Normalised Laplacian eigenvalue evolution")
    ax.legend(
        fontsize=8.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0, frameon=False
    )
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Covariance eigenvalues 1/(lambda + eps)
    ax = axes[1]
    cov_evals = 1.0 / (all_evals + EPS)
    for i in range(K_SHOW):
        color = cmap(i / K_SHOW)
        ax.plot(
            valid_epochs,
            cov_evals[:, i],
            "-o",
            markersize=2.5,
            color=color,
            label=f"$1/(\\lambda_{{{i + 1}}}+\\epsilon)$",
            linewidth=1.2,
        )
    shade_grokking_window(ax)
    ax.set_ylabel("Covariance eigenvalue $1/(\\lambda_k + \\epsilon)$")
    ax.set_title(f"Induced covariance spectrum ($\\epsilon={EPS}$)")
    ax.set_yscale("log")
    ax.legend(
        fontsize=8.5, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0, frameon=False
    )
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Validation accuracy
    ax = axes[2]
    ax.plot(meta_epochs, val_accs, color=VALIDATION_COLOR, linewidth=1.8, label="validation accuracy")
    shade_grokking_window(ax)
    ax.set_ylabel("Val accuracy")
    ax.set_xlabel("Training epoch")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.10, right=0.76, bottom=0.08, top=0.95, hspace=0.36)
    fig.savefig(FIG_DIR / "fig_eigenvalue_evolution.pdf")
    fig.savefig(FIG_DIR / "fig_eigenvalue_evolution.png")
    print("Saved: fig_eigenvalue_evolution.pdf/.png")
    plt.close(fig)

    for i, ep in enumerate(valid_epochs):
        gap = all_evals[i, 1] - all_evals[i, 0] if K_SHOW > 1 else 0.0
        print(
            f"  epoch {ep:>6d}: lambda_1={all_evals[i, 0]:.4f}, lambda_2={all_evals[i, 1]:.4f}, gap={gap:.4f}"
        )

    print("eigenvalue figure exists", (FIG_DIR / "fig_eigenvalue_evolution.pdf").exists())

    print("valid epochs", len(valid_epochs), "expected", len(EPOCHS))

    print("spectrum shape", all_evals.shape)
    print("expected shape", (len(valid_epochs), K_SHOW))

    print("laplacian eigenvalue range", float(np.min(all_evals)), float(np.max(all_evals)))
    print("covariance eigenvalue range", float(np.min(cov_evals)), float(np.max(cov_evals)))

    print("early eigenvalues", valid_epochs[0], all_evals[0, :5])
    print("late eigenvalues", valid_epochs[-1], all_evals[-1, :5])

    qa_lam = np.array([0.0, 0.5, 2.0])
    qa_res = 1.0 / (qa_lam + EPS)
    print("lambda", qa_lam)
    print("resolvent", qa_res)


if __name__ == "__main__":
    main()
