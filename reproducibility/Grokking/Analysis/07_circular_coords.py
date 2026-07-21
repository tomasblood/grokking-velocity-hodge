# Databricks notebook source

# MAGIC %md
# MAGIC # Grokking 07: Circular Coordinate Recovery
# MAGIC
# MAGIC Inputs:
# MAGIC - Checkpoint activation snapshots
# MAGIC - Modular addition labels
# MAGIC
# MAGIC Outputs:
# MAGIC - Circular coordinate recovery metrics
# MAGIC - Figure: fig_circular_coords.pdf
# MAGIC - JSON: circular_results.json

import gc
import json
import os
import sys

import numpy as np
import diffusion_geometry as dg
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# COMMAND ----------

sys.path.insert(0, os.environ["THESIS_SHARED_DIR"])
from runtime import (
    ACCENT_COLOR,
    GREY_COLOR,
    MAIN_COLOR,
    VALIDATION_COLOR,
    circular_correlation,
    configure_grokking_runtime,
    dg_circular_coordinate,
    dg_circular_coordinate_from_rep,
    dimmed_phase_cmap,
    dominant_fourier_frequency,
    fourier_circular_coordinate,
    load_training_meta,
    notebook_param,
    set_paper_style,
    shade_grokking_window,
    write_json,
)

GROKKING = configure_grokking_runtime()
ROOT = GROKKING.root
ACT_DIR = GROKKING.activation_dir
FIG_DIR = GROKKING.figure_dir
PHASE_CMAP_NAME = notebook_param("GROKKING_PHASE_CMAP", "hsv").strip() or "hsv"
PHASE_CMAP_SATURATION = float(notebook_param("GROKKING_PHASE_SATURATION", "0.58"))
PHASE_CMAP_VALUE = float(notebook_param("GROKKING_PHASE_VALUE", "0.92"))
OUT_DIR = GROKKING.result_dir("grokking_circular_coords")
set_paper_style()
print(f"ROOT: {ROOT}")

P = 113
EPOCHS = list(range(0, 25001, 500))
KNN = 15
N_BASIS = 30
PCA_DIM = 10
N_EIGPAIRS = 15


PHASE_CMAP = dimmed_phase_cmap(PHASE_CMAP_NAME, PHASE_CMAP_SATURATION, PHASE_CMAP_VALUE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load labels and training curve

# COMMAND ----------

gt = np.load(ACT_DIR / "gt_labels.npy")

# Load training curves
training = load_training_meta(ACT_DIR)
saved_epochs = training["saved_epochs"]
val_accs = training["val_accs"]

results = []

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Output: circular_results.json

# COMMAND ----------

# DBTITLE 1,circular_results.json
for ep in EPOCHS:
    act_path = ACT_DIR / f"act_{ep}.npy"
    assert act_path.exists(), f"Missing required activation snapshot: {act_path}"

    X = np.load(act_path).astype(np.float64)

    # Fourier analysis
    k_dom, powers = dominant_fourier_frequency(X, gt, p=P)
    top_power = powers[np.argmax(powers)]
    total_power = powers[1:].sum()
    dom_frac = top_power / max(total_power, 1e-12)

    # Fourier supervised circular coordinate
    theta_fourier = fourier_circular_coordinate(X, gt, k_dom, p=P)
    theta_true_k = (gt * k_dom * 2 * np.pi / P) % (2 * np.pi)
    corr_fourier = circular_correlation(theta_fourier, theta_true_k)

    # DG unsupervised
    theta_dg, corr_dg, pair_dg, k_dg = dg_circular_coordinate(
        X,
        gt,
        k_dom,
        PCA,
        dg,
        p=P,
        pca_dim=PCA_DIM,
        knn=KNN,
        n_basis=N_BASIS,
        n_eigpairs=N_EIGPAIRS,
    )

    # Validation accuracy (nearest saved epoch)
    ep_idx = min(range(len(saved_epochs)), key=lambda i: abs(saved_epochs[i] - ep))
    va = val_accs[ep_idx]

    rec = {
        "epoch": ep,
        "k_dominant": k_dom,
        "dom_frac": round(dom_frac, 4),
        "corr_fourier": round(corr_fourier, 4),
        "corr_dg": round(corr_dg, 4),
        "dg_pair": list(pair_dg),
        "dg_freq": k_dg,
        "val_acc": round(va, 4),
    }
    results.append(rec)
    print(
        f"  Epoch {ep:5d}: k_dom={k_dom:2d} "
        f"fourier={corr_fourier:.3f} dg={corr_dg:.3f} "
        f"(phi_{pair_dg[0]},phi_{pair_dg[1]}) "
        f"val_acc={va:.3f}"
    )

# --- Save JSON
write_json(OUT_DIR / "circular_results.json", results)
print(f"\nSaved: {OUT_DIR / 'circular_results.json'}")

# --- Publication figure (2x2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Circular coordinate recovery time series

# COMMAND ----------

epochs_arr = [r["epoch"] for r in results]
corr_f = [r["corr_fourier"] for r in results]
corr_d = [r["corr_dg"] for r in results]
va_arr = [r["val_acc"] for r in results]
k_arr = [r["k_dominant"] for r in results]

fig, axes = plt.subplots(2, 2, figsize=(8.8, 9.4))

# Panel 0 0 circular correlation over time
ax = axes[0, 0]
ax.plot(epochs_arr, corr_f, "o-", color=MAIN_COLOR, markersize=3, linewidth=2.0, label="Fourier-supervised")
ax.plot(epochs_arr, corr_d, "s-", color=ACCENT_COLOR, markersize=3, linewidth=2.0, label="DG unsupervised")
ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
shade_grokking_window(ax)
ax.set_ylabel("Circular correlation")
ax.set_xlabel("Epoch")
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, epochs_arr[-1] + 900)
ax.legend(loc="lower right", fontsize=10)
ax.set_title("Circular coordinate recovery during grokking")

# Overlay val accuracy on twin axis
ax2 = ax.twinx()
ax2.fill_between(epochs_arr, va_arr, alpha=0.08, color=VALIDATION_COLOR)
ax2.plot(epochs_arr, va_arr, color=VALIDATION_COLOR, linewidth=1.0, alpha=0.7)
ax2.set_ylabel("Val accuracy", color=GREY_COLOR)
ax2.tick_params(axis="y", colors=GREY_COLOR)
ax2.set_ylim(-0.05, 1.3)

# Panel 0 1 dominant Fourier frequency over time
ax = axes[0, 1]
ax.plot(epochs_arr, k_arr, "o-", color=ACCENT_COLOR, markersize=3, linewidth=2.0)
shade_grokking_window(ax)
ax.set_ylabel("Dominant frequency k")
ax.set_xlabel("Epoch")
ax.set_title("Dominant Fourier frequency is stable post-transition")

# Panels 1 0 and 1 1 PCA scatter at epoch 2500 and 25000
# coloured by DG circular coordinate

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Figure: fig_circular_coords.pdf

# COMMAND ----------

# DBTITLE 1,fig_circular_coords.pdf
for col, ep_show in enumerate([2500, 25000]):
    ax = axes[1, col]
    act_path = ACT_DIR / f"act_{ep_show}.npy"
    X_show = np.load(act_path).astype(np.float64)
    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X_show)

    k_dom_show, _ = dominant_fourier_frequency(X_show, gt, p=P)
    theta_dg_show, corr_show, pair_show, k_show = dg_circular_coordinate_from_rep(
        X_2d,
        gt,
        k_dom_show,
        dg,
        p=P,
        knn=KNN,
        n_basis=N_BASIS,
        n_eigpairs=N_EIGPAIRS,
    )
    sc = ax.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=theta_dg_show,
        cmap=PHASE_CMAP,
        s=10,
        alpha=0.7,
    )
    ax.set_title(
        f"Epoch {ep_show} (DG-on-PCA corr={corr_show:.2f})\n"
        f"phi_{pair_show[0]}, phi_{pair_show[1]} at k={k_show}"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, label="theta (rad)")

fig.suptitle("DG circular-coordinate recovery during grokking", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG_DIR / "fig_circular_coords.pdf")
fig.savefig(FIG_DIR / "fig_circular_coords.png")
plt.close(fig)
print(f"Saved: {FIG_DIR / 'fig_circular_coords.pdf'}")

gc.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## QA

# COMMAND ----------

# DBTITLE 1,saved files
qa_path = OUT_DIR / "circular_results.json"
print("json exists", qa_path.exists())
print("circular coords figure exists", (FIG_DIR / "fig_circular_coords.pdf").exists())
with open(qa_path) as f:
    qa_rows = json.load(f)

# COMMAND ----------

# DBTITLE 1,circular row count
print("circular rows", len(qa_rows), "expected", len(EPOCHS))

# COMMAND ----------

# DBTITLE 1,correlation range
print("fourier corr range", min(row["corr_fourier"] for row in qa_rows), max(row["corr_fourier"] for row in qa_rows))
print("dg corr range", min(row["corr_dg"] for row in qa_rows), max(row["corr_dg"] for row in qa_rows))

# COMMAND ----------

# DBTITLE 1,best recovery epoch
best_fourier = max(qa_rows, key=lambda row: row["corr_fourier"])
best_dg = max(qa_rows, key=lambda row: row["corr_dg"])
print("best fourier", best_fourier["epoch"], best_fourier["corr_fourier"])
print("best dg", best_dg["epoch"], best_dg["corr_dg"])

# COMMAND ----------

# DBTITLE 1,known circle coordinate check
qa_theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
print("self circular correlation", circular_correlation(qa_theta, qa_theta))
print("shifted circular correlation", circular_correlation(qa_theta, (qa_theta + 0.5) % (2.0 * np.pi)))
