"""Compute effective-dimension, PCA, and Fourier diagnostics."""

import gc

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grokking_velocity_hodge.config import ExperimentConfig
from grokking_velocity_hodge.runtime import (
    ACCENT_COLOR,
    GREY_COLOR,
    MAIN_COLOR,
    best_fourier_ridge_frequency,
    configure_grokking_runtime,
    dimmed_phase_cmap,
    fourier_ridge_projection,
    load_training_meta,
    notebook_param,
    set_paper_style,
    shade_grokking_window,
)


def main() -> None:
    GROKKING = configure_grokking_runtime()
    CONFIG = ExperimentConfig.from_environment()
    ROOT = GROKKING.root
    ACT_DIR = GROKKING.activation_dir
    FIG_DIR = GROKKING.figure_dir
    PHASE_CMAP_NAME = notebook_param("GROKKING_PHASE_CMAP", "hsv").strip() or "hsv"
    PHASE_CMAP_SATURATION = float(notebook_param("GROKKING_PHASE_SATURATION", "0.58"))
    PHASE_CMAP_VALUE = float(notebook_param("GROKKING_PHASE_VALUE", "0.92"))
    set_paper_style()
    print(f"ROOT: {ROOT}")

    P = CONFIG.modulus
    training = load_training_meta(ACT_DIR)
    EPOCHS = CONFIG.checkpoint_epochs(training)
    PCA_EPOCHS = [epoch for epoch in [0, 2500, 3000, 7500, 10000, EPOCHS[-1]] if epoch in EPOCHS]

    PHASE_CMAP = dimmed_phase_cmap(PHASE_CMAP_NAME, PHASE_CMAP_SATURATION, PHASE_CMAP_VALUE)

    eff_dims = []
    for ep in EPOCHS:
        p = ACT_DIR / f"act_{ep}.npy"
        assert p.exists(), f"Missing required activation snapshot: {p}"
        X = np.load(p).astype(np.float64)
        cov = np.cov(X.T)
        ev = np.maximum(np.linalg.eigvalsh(cov), 0)
        s1, s2 = ev.sum(), (ev**2).sum()
        d = s1**2 / s2 if s2 > 0 else 0
        eff_dims.append({"epoch": ep, "d_eff": round(float(d), 2)})

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    eff_epochs = [r["epoch"] for r in eff_dims]
    eff_values = [r["d_eff"] for r in eff_dims]
    shade_grokking_window(ax, label=True)
    ax.plot(eff_epochs, eff_values, color=MAIN_COLOR, lw=2.2)
    ax.scatter(
        [eff_epochs[0], eff_epochs[-1]],
        [eff_values[0], eff_values[-1]],
        color=[GREY_COLOR, ACCENT_COLOR],
        s=36,
        zorder=3,
    )
    ax.annotate(
        f"{eff_values[0]:.1f}",
        xy=(eff_epochs[0], eff_values[0]),
        xytext=(8, -2),
        textcoords="offset points",
        color=GREY_COLOR,
        fontsize=10,
        va="top",
    )
    ax.annotate(
        f"{eff_values[-1]:.1f}",
        xy=(eff_epochs[-1], eff_values[-1]),
        xytext=(-8, 8),
        textcoords="offset points",
        color=ACCENT_COLOR,
        fontsize=10,
        ha="right",
    )
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Participation ratio")
    ax.set_title("Hidden activations collapse to a low-dimensional representation")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_effective_dim.pdf")
    fig.savefig(FIG_DIR / "fig_effective_dim.png")
    plt.close(fig)
    print(f"d_eff: {eff_dims[0]['d_eff']} -> {eff_dims[-1]['d_eff']}")
    gc.collect()

    gt = np.load(ACT_DIR / "gt_labels.npy")
    training = load_training_meta(ACT_DIR)
    val_accs = training.get("val_accs", [])
    saved_epochs = training.get("saved_epochs", training.get("epochs", []))

    final_epoch = EPOCHS[-1]
    X_late = np.load(ACT_DIR / f"act_{final_epoch}.npy").astype(np.float64)
    best_k, best_r2, scored_freqs = best_fourier_ridge_frequency(X_late, gt, p=P)
    top_freqs = ", ".join(f"k={k} (R2={r2:.3f})" for k, r2 in scored_freqs[:5])
    print(f"Dominant ridge frequency at epoch {final_epoch}: k={best_k}, R2={best_r2:.4f}")
    print(f"Top frequencies: {top_freqs}")

    fourier_label = (best_k * gt) % P
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 11.2))
    for idx, ep in enumerate(PCA_EPOCHS):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        p = ACT_DIR / f"act_{ep}.npy"
        assert p.exists(), f"Missing required activation snapshot: {p}"

        X = np.load(p).astype(np.float64)
        z_cos, z_sin, r2 = fourier_ridge_projection(X, gt, best_k, p=P)
        ep_idx = min(range(len(saved_epochs)), key=lambda i: abs(saved_epochs[i] - ep))
        va = val_accs[ep_idx]

        sc = ax.scatter(
            z_cos,
            z_sin,
            c=fourier_label,
            cmap=PHASE_CMAP,
            s=12,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(f"Epoch {ep:,}  (R2={r2:.2f}, val={va:.0%})", fontsize=12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#888888")

    fig.suptitle(
        f"Fourier circle emergence during grokking\n"
        f"(ridge projection onto cos/sin of frequency k = {best_k}, coloured by Fourier phase)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.93])
    cax = fig.add_axes([0.22, 0.014, 0.56, 0.014])
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Fourier phase (k * label mod P)", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    fig.savefig(FIG_DIR / "fig_fourier_ridge_pca.pdf")
    fig.savefig(FIG_DIR / "fig_fourier_ridge_pca.png")
    plt.close(fig)
    print(f"Saved Fourier ridge PCA grid (k={best_k})")
    gc.collect()

    fig, axes = plt.subplots(3, 2, figsize=(8.8, 11.2))
    for idx, ep in enumerate(PCA_EPOCHS):
        p = ACT_DIR / f"act_{ep}.npy"
        assert p.exists(), f"Missing required activation snapshot: {p}"
        X = np.load(p).astype(np.float64)
        pca = PCA(n_components=2, random_state=42).fit(X)
        Z = pca.transform(X)
        ax = axes[idx // 2, idx % 2]
        ax.scatter(Z[:, 0], Z[:, 1], c=gt[: len(Z)], cmap=PHASE_CMAP, s=8, alpha=0.8)
        ax.set_title(f"Epoch {ep}", fontsize=12)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("PCA of Grokking Activations (coloured by label)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_pca_grid.pdf")
    fig.savefig(FIG_DIR / "fig_pca_grid.png")
    plt.close(fig)
    print("Saved PCA grid")

    print("effective dimension figure exists", (FIG_DIR / "fig_effective_dim.pdf").exists())
    print("fourier ridge figure exists", (FIG_DIR / "fig_fourier_ridge_pca.pdf").exists())
    print("pca grid figure exists", (FIG_DIR / "fig_pca_grid.pdf").exists())

    eff_values = np.array([row["d_eff"] for row in eff_dims], dtype=float)
    print("effective dimension rows", len(eff_dims), "expected", len(EPOCHS))
    print("first last", eff_dims[0]["d_eff"], eff_dims[-1]["d_eff"])
    print("range", float(eff_values.min()), float(eff_values.max()))

    ratio = eff_dims[-1]["d_eff"] / eff_dims[0]["d_eff"]
    min_row = min(eff_dims, key=lambda row: row["d_eff"])
    print("final over initial", ratio)
    print("minimum effective dimension", min_row["epoch"], min_row["d_eff"])

    print("best ridge frequency", best_k)
    print("best ridge r2", best_r2)
    print("top frequencies", scored_freqs[:5])

    x_first = np.load(ACT_DIR / f"act_{EPOCHS[0]}.npy")
    x_last = np.load(ACT_DIR / f"act_{EPOCHS[-1]}.npy")
    print("first activation shape", x_first.shape)
    print("last activation shape", x_last.shape)

    qa_x = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    qa_cov = np.cov(qa_x.T)
    qa_eigs = np.linalg.eigvalsh(qa_cov)
    qa_pr = (qa_eigs.sum() ** 2) / np.sum(qa_eigs**2)
    print("toy eigenvalues", qa_eigs)
    print("toy participation ratio", float(qa_pr))


if __name__ == "__main__":
    main()
