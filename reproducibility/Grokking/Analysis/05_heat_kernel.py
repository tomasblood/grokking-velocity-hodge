"""Compute global heat-kernel Bures--Wasserstein distances."""

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
    VALIDATION_COLOR,
    build_normalised_laplacian,
    bw_distance,
    configure_grokking_runtime,
    heat_kernel_in_reference_basis,
    laplacian_spectrum,
    load_training_meta,
    set_paper_style,
    shade_grokking_window,
    write_json,
)


def main() -> None:
    GROKKING = configure_grokking_runtime()
    CONFIG = ExperimentConfig.from_environment()
    ROOT = GROKKING.root
    ACT_DIR = GROKKING.activation_dir
    FIG_DIR = GROKKING.figure_dir
    OUT_DIR = GROKKING.result_dir("grokking_heat_kernel")
    set_paper_style()
    print(f"ROOT: {ROOT}")

    PCA_DIM = CONFIG.pca_dim
    PCA_SOLVER = CONFIG.pca_solver
    KNN = CONFIG.knn
    K_SPEC = CONFIG.spectral_components

    training = load_training_meta(ACT_DIR)
    EPOCHS = CONFIG.checkpoint_epochs(training)
    val_accs = training.get("val_accs", [])
    meta_epochs = training.get("epochs", training.get("saved_epochs", list(range(len(val_accs)))))
    print(f"Training metadata: {len(val_accs)} entries")

    TAU_VALUES = list(CONFIG.heat_scales)
    scale_names = {0.1: "local", 1.0: "meso", 10.0: "global"}
    TAU_LABELS = [f"{scale_names.get(tau, 'scale')}, $\\tau={tau:g}$" for tau in TAU_VALUES]
    palette = [MAIN_COLOR, ACCENT_COLOR, GREY_COLOR]
    TAU_COLORS = [palette[index % len(palette)] for index in range(len(TAU_VALUES))]

    # --- Reference basis from epoch 0
    path_ref = ACT_DIR / "act_0.npy"
    X_ref = np.load(path_ref).astype(np.float64)
    pca_ref = PCA(n_components=PCA_DIM, svd_solver=PCA_SOLVER)
    X_ref_red = pca_ref.fit_transform(X_ref)
    L_ref = build_normalised_laplacian(X_ref_red, k=KNN)
    evals_ref, evecs_ref = laplacian_spectrum(L_ref, k=K_SPEC)
    ref_basis = evecs_ref
    print(f"Reference basis from epoch 0: {ref_basis.shape}")

    # --- Compute heat kernel covariances at all scales
    # hk_covs[tau][epoch] = H_ref matrix
    hk_covs = {tau: {} for tau in TAU_VALUES}
    valid_epochs = []

    for ep in EPOCHS:
        path = ACT_DIR / f"act_{ep}.npy"
        assert path.exists(), f"Missing required activation snapshot: {path}"
        X = np.load(path).astype(np.float64)
        pca = PCA(n_components=PCA_DIM, svd_solver=PCA_SOLVER)
        X_red = pca.fit_transform(X)

        L = build_normalised_laplacian(X_red, k=KNN)
        evals, evecs = laplacian_spectrum(L, k=K_SPEC)

        for tau in TAU_VALUES:
            H = heat_kernel_in_reference_basis(evals, evecs, ref_basis, tau)
            hk_covs[tau][ep] = H

        valid_epochs.append(ep)
        print(f"  epoch {ep:>6d}: done (3 scales)")
        gc.collect()

    print(f"\nComputed heat kernels for {len(valid_epochs)} epochs at {len(TAU_VALUES)} scales")

    # bw_series[tau] stores midpoint epoch and distance pairs
    bw_series = {tau: [] for tau in TAU_VALUES}

    for tau in TAU_VALUES:
        for i in range(1, len(valid_epochs)):
            ep_a = valid_epochs[i - 1]
            ep_b = valid_epochs[i]
            d = bw_distance(hk_covs[tau][ep_a], hk_covs[tau][ep_b])
            mid = (ep_a + ep_b) / 2.0
            bw_series[tau].append((mid, d))
        print(f"  tau={tau}: {len(bw_series[tau])} consecutive BW distances computed")

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})

    # --- Panel 1: BW distances at all 3 scales
    ax = axes[0]
    for idx, tau in enumerate(TAU_VALUES):
        mids = [p[0] for p in bw_series[tau]]
        dists = [p[1] for p in bw_series[tau]]
        ax.plot(mids, dists, "-o", markersize=3, color=TAU_COLORS[idx], linewidth=2.0, label=TAU_LABELS[idx])
        label_offsets = [0.10, -0.02, -0.10]
        ax.text(
            mids[-1] + 450,
            dists[-1] + label_offsets[idx],
            TAU_LABELS[idx],
            color=TAU_COLORS[idx],
            fontsize=10,
            va="center",
        )

    shade_grokking_window(ax, label=True)
    ax.set_ylabel("$d_{BW}$ (consecutive)")
    ax.set_title("Heat-kernel BW distances across diffusion scales")
    ax.set_xlim(0, max(mids) + 2200)
    ax.legend(fontsize=10, loc="upper right")

    # --- Annotate scale ratio
    # Compare peak BW at local vs global
    for idx, tau in enumerate(TAU_VALUES):
        dists = [p[1] for p in bw_series[tau]]
        peak = max(dists)
        peak_mid = [p[0] for p in bw_series[tau]][np.argmax(dists)]
        ax.annotate(
            f"peak={peak:.3f}",
            xy=(peak_mid, peak),
            fontsize=9,
            color=TAU_COLORS[idx],
            textcoords="offset points",
            xytext=(5, 5),
        )

    # --- Panel 2: Val accuracy
    ax = axes[1]
    ax.plot(meta_epochs, val_accs, color=VALIDATION_COLOR, linewidth=1.8, label="validation accuracy")
    shade_grokking_window(ax)
    ax.set_ylabel("Val accuracy")
    ax.set_xlabel("Training epoch")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(mids) + 2200)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_heat_kernel_bw.pdf")
    fig.savefig(FIG_DIR / "fig_heat_kernel_bw.png")
    print("Saved: fig_heat_kernel_bw.pdf/.png")
    plt.close(fig)

    results = {
        "valid_epochs": valid_epochs,
        "tau_values": TAU_VALUES,
        "bw_series": {},
    }
    for tau in TAU_VALUES:
        results["bw_series"][str(tau)] = {
            "midpoint_epochs": [float(p[0]) for p in bw_series[tau]],
            "distances": [float(p[1]) for p in bw_series[tau]],
        }

    out_json = OUT_DIR / "heat_kernel_bw_results.json"
    write_json(out_json, results)
    print(f"Saved: {out_json}")

    print("json exists", out_json.exists())
    print("heat kernel figure exists", (FIG_DIR / "fig_heat_kernel_bw.pdf").exists())

    print("expected tau", TAU_VALUES)
    print("tau keys", sorted(bw_series.keys()))

    for tau, rows in bw_series.items():
        print("tau", tau, "rows", len(rows), "expected", len(valid_epochs) - 1)

    for tau, rows in bw_series.items():
        distances = np.array([distance for _, distance in rows], dtype=float)
        print(
            "tau",
            tau,
            "min mean max",
            float(distances.min()),
            float(distances.mean()),
            float(distances.max()),
        )

    qa_evals = np.array([0.0, 0.5, 1.0])
    qa_basis = np.eye(3)
    qa_h1 = heat_kernel_in_reference_basis(qa_evals, qa_basis, qa_basis, tau=1.0)
    qa_h2 = heat_kernel_in_reference_basis(qa_evals, qa_basis, qa_basis, tau=1.0)
    print("same spectrum bw", bw_distance(qa_h1, qa_h2))
    print("heat diagonal", np.diag(qa_h1))

    qa_h_small = heat_kernel_in_reference_basis(qa_evals, qa_basis, qa_basis, tau=0.1)
    qa_h_large = heat_kernel_in_reference_basis(qa_evals, qa_basis, qa_basis, tau=2.0)
    print("tau 0.1 diag", np.diag(qa_h_small))
    print("tau 2.0 diag", np.diag(qa_h_large))


if __name__ == "__main__":
    main()
