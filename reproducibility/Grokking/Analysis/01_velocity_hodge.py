# Databricks notebook source

# MAGIC %md
# MAGIC # Grokking 01: Diffusion Geometry Velocity Hodge Decomposition
# MAGIC
# MAGIC Inputs:
# MAGIC - Checkpoint activation snapshots
# MAGIC - Training accuracy metadata
# MAGIC
# MAGIC Outputs:
# MAGIC - Hodge exact/coexact/harmonic fractions
# MAGIC - Figure: fig_dg_velocity_hodge.pdf
# MAGIC - Figure: fig_dg_velocity_quiver.pdf
# MAGIC - JSON: velocity_hodge.json

# COMMAND ----------




# MAGIC %pip install -q "git+https://github.com/Iolo-Jones/DiffusionGeometry.git@f5dc795557d07b32795c0bb6bedf465246d999eb" scikit-learn scipy matplotlib

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os, sys, gc, json
import numpy as np
from pathlib import Path
import diffusion_geometry as dg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# COMMAND ----------

import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["THESIS_SHARED_DIR"])
from runtime import *

GROKKING = configure_grokking_runtime()
ROOT = GROKKING.root
ACT_DIR = GROKKING.activation_dir
FIG_DIR = GROKKING.figure_dir
OUT_DIR = GROKKING.result_dir("grokking_dg_velocity_hodge")
set_paper_style()
print(f"ROOT: {ROOT}")

EPOCHS = list(range(0, 25001, 500))
PCA_DIM = 10
PCA_SOLVER = "full"
KNN = 15
N_BASIS = 50

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load training curve

# COMMAND ----------

# Load training curves
training = load_training_meta(ACT_DIR)
saved_epochs = training["saved_epochs"]
val_accs = training["val_accs"]

# --- Part 1: Hodge fractions across all epochs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Compute Hodge fractions

# COMMAND ----------

results = []
for i in range(len(EPOCHS) - 1):
    ep_a, ep_b = EPOCHS[i], EPOCHS[i + 1]
    path_a = ACT_DIR / f"act_{ep_a}.npy"
    path_b = ACT_DIR / f"act_{ep_b}.npy"
    assert path_a.exists() and path_b.exists(), f"Missing activation pair: {path_a}, {path_b}"

    X_a = np.load(path_a).astype(np.float64)
    X_b = np.load(path_b).astype(np.float64)
    V = X_b - X_a

    decomp = hodge_decompose_velocity(
        X_a,
        V,
        PCA,
        dg,
        pca_dim=PCA_DIM,
        pca_solver=PCA_SOLVER,
        knn=KNN,
        n_basis=N_BASIS,
    )

    # Validation accuracy at midpoint
    mid = (ep_a + ep_b) / 2
    ep_idx = min(range(len(saved_epochs)), key=lambda j: abs(saved_epochs[j] - mid))
    va = val_accs[ep_idx]

    rec = {
        "epoch_a": ep_a, "epoch_b": ep_b, "midpoint": mid,
        "exact": round(decomp["exact"], 4),
        "coexact": round(decomp["coexact"], 4),
        "harmonic": round(decomp["harmonic"], 4),
        "total_energy": round(decomp["total_energy"], 4),
        "val_acc": round(va, 4),
    }
    results.append(rec)
    print(
        f"  {ep_a:5d}->{ep_b:5d}: "
        f"exact={decomp['exact']:.3f} coexact={decomp['coexact']:.3f} "
        f"harmonic={decomp['harmonic']:.3f}  energy={decomp['total_energy']:.2e}"
    )

# Save JSON

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Output: velocity_hodge.json

# COMMAND ----------

# DBTITLE 1,velocity_hodge.json
summary = {
    "config": {"pca_dim": PCA_DIM, "knn": KNN, "n_basis": N_BASIS},
    "pairs": results,
    "mean_exact": round(float(np.mean([r["exact"] for r in results])), 4),
    "mean_coexact": round(float(np.mean([r["coexact"] for r in results])), 4),
    "mean_harmonic": round(float(np.mean([r["harmonic"] for r in results])), 4),
}
write_json(OUT_DIR / "velocity_hodge.json", summary)
print(f"\nSaved: {OUT_DIR / 'velocity_hodge.json'}")

# --- Part 2: Smoothed line chart

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Figure: fig_dg_velocity_hodge.pdf

# COMMAND ----------

# DBTITLE 1,fig_dg_velocity_hodge.pdf
from matplotlib.ticker import MultipleLocator
midpoints = np.array([r["midpoint"] for r in results])
exact = np.array([r["exact"] for r in results])
coexact = np.array([r["coexact"] for r in results])
harmonic = np.array([r["harmonic"] for r in results])
va_arr = np.array([r["val_acc"] for r in results])

# Clip negatives and renormalise
exact_c = np.clip(exact, 0, None)
coexact_c = np.clip(coexact, 0, None)
harmonic_c = np.clip(harmonic, 0, None)
total = exact_c + coexact_c + harmonic_c
exact_n = exact_c / total; coexact_n = coexact_c / total; harmonic_n = harmonic_c / total

exact_s = edge_padded_moving_average(exact_n)
coexact_s = edge_padded_moving_average(coexact_n)
harmonic_s = edge_padded_moving_average(harmonic_n)
ts = exact_s + coexact_s + harmonic_s
exact_s /= ts; coexact_s /= ts; harmonic_s /= ts

c_ex, c_co, c_ha = MAIN_COLOR, ACCENT_COLOR, SECONDARY_COLOR
fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.8), height_ratios=[3, 1],
                          sharex=True, gridspec_kw={'hspace': 0.08})
ax = axes[0]
ax.scatter(midpoints, exact_n, s=18, color=c_ex, alpha=0.25, zorder=2, edgecolors='none')
ax.scatter(midpoints, coexact_n, s=18, color=c_co, alpha=0.25, zorder=2, edgecolors='none')
ax.scatter(midpoints, harmonic_n, s=18, color=c_ha, alpha=0.25, zorder=2, edgecolors='none')
ax.plot(midpoints, exact_s, color=c_ex, linewidth=2.5, zorder=3)
ax.plot(midpoints, coexact_s, color=c_co, linewidth=2.5, zorder=3)
ax.plot(midpoints, harmonic_s, color=c_ha, linewidth=2.5, zorder=3)
shade_grokking_window(ax, label=True)
label_x = 25300
ax.text(label_x, exact_s[-1], 'exact', color=c_ex, fontsize=11, va='center')
ax.text(label_x, coexact_s[-1], 'coexact', color=c_co, fontsize=11, va='center')
ax.text(label_x, harmonic_s[-1], 'harmonic', color=c_ha, fontsize=11, va='center')
ax.set_ylabel('Energy fraction', fontsize=12)
ax.set_ylim(0, 1); ax.set_xlim(0, 27500)
ax.legend(loc='upper left', fontsize=10)
ax.yaxis.set_major_locator(MultipleLocator(0.2))

ax2 = axes[1]
ax2.plot(midpoints, va_arr, color=VALIDATION_COLOR, linewidth=1.8)
shade_grokking_window(ax2)
ax2.set_ylabel('Val acc', fontsize=12); ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_xlim(0, 27500)
ax2.set_ylim(-0.05, 1.05)
ax2.xaxis.set_major_locator(MultipleLocator(5000))
fig.align_ylabels()
fig.subplots_adjust(hspace=0.08)
fig.savefig(FIG_DIR / "fig_dg_velocity_hodge.pdf", bbox_inches='tight')
fig.savefig(FIG_DIR / "fig_dg_velocity_hodge.png", bbox_inches='tight')
if "display" in globals():
    display(fig)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'fig_dg_velocity_hodge.pdf'}")

# --- Part 3: Quiver plots at 3 key epochs with subsampled DG Hodge

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Figure: fig_dg_velocity_quiver.pdf

# COMMAND ----------

# DBTITLE 1,fig_dg_velocity_quiver.pdf
PAIRS = [(1500, 2000), (2500, 3000), (4000, 4500)]
ROW_LABELS = ["Pre-grokking\n(1500 \u2192 2000)",
              "Transition\n(2500 \u2192 3000)",
              "Post-grokking\n(4000 \u2192 4500)"]
COL_LABELS = ["Total", "Exact (gradient)", "Coexact (curl)", "Harmonic"]
COLORS_Q = [GREY_COLOR, MAIN_COLOR, ACCENT_COLOR, SECONDARY_COLOR]
N_SHOW = 60

fig, axes_q = plt.subplots(4, 3, figsize=(8.8, 10.8))

for epoch_col, ((ep_a, ep_b), row_label) in enumerate(zip(PAIRS, ROW_LABELS)):
    X_a = np.load(ACT_DIR / f"act_{ep_a}.npy").astype(np.float64)
    X_b = np.load(ACT_DIR / f"act_{ep_b}.npy").astype(np.float64)

    # PCA-10 reduction
    pca10 = PCA(n_components=PCA_DIM, svd_solver=PCA_SOLVER)
    X_pca = pca10.fit_transform(X_a)
    v_full = pca10.transform(X_b) - X_pca

    # DG Hodge decomposition with to_ambient()
    model = dg.DiffusionGeometry.from_point_cloud(
        X_pca.astype(np.float64), knn=KNN, n_basis=N_BASIS,
    )
    omega = model.form(v_full.astype(np.float64), degree=1)
    f_pot, g_pot, h_form = omega.hodge_decomposition()
    exact_1 = f_pot.d()
    coexact_1 = g_pot.codifferential()

    v_exact = exact_1.to_ambient()
    v_coexact = coexact_1.to_ambient()
    v_harmonic = h_form.to_ambient()

    # Energy fractions (normalised)
    e_ex = float(exact_1.norm()**2)
    e_co = float(coexact_1.norm()**2)
    e_ha = float(h_form.norm()**2)
    e_sum = e_ex + e_co + e_ha
    fracs = [1.0, e_ex/e_sum, e_co/e_sum, e_ha/e_sum]

    # PCA-2 for visualisation
    pca2 = PCA(n_components=2, svd_solver=PCA_SOLVER)
    pos = pca2.fit_transform(X_pca)
    fields = [v_full @ pca2.components_.T,
              v_exact @ pca2.components_.T,
              v_coexact @ pca2.components_.T,
              v_harmonic @ pca2.components_.T]

    idx = farthest_point_sample(pos, N_SHOW)

    for component_row, (V_2d, col_label, qcol, frac) in enumerate(
        zip(fields, COL_LABELS, COLORS_Q, fracs)
    ):
        ax = axes_q[component_row, epoch_col]
        ax.scatter(pos[:, 0], pos[:, 1], s=2, c='#d8d8d8', alpha=0.55, zorder=1)
        norms = np.linalg.norm(V_2d[idx], axis=1)
        p90 = np.percentile(norms[norms > 0], 90) if np.any(norms > 0) else 1.0
        ax.quiver(pos[idx, 0], pos[idx, 1], V_2d[idx, 0], V_2d[idx, 1],
                  color=qcol, alpha=0.75, scale=p90 * 8,
                  width=0.005, headwidth=3.5, headlength=4, zorder=2)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color('#cccccc')
        if component_row == 0:
            ax.set_title(row_label, fontsize=11, fontweight='bold', pad=6)
        if epoch_col == 0:
            ax.set_ylabel(col_label, fontsize=11, fontweight='bold')
        if component_row > 0:
            ax.text(0.96, 0.04, f'{frac*100:.0f}%', transform=ax.transAxes,
                    fontsize=10, fontweight='bold', ha='right', va='bottom',
                    color=qcol, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=qcol, alpha=0.85))
    print(f"  Quiver {ep_a}->{ep_b}: exact={fracs[1]:.1%} coexact={fracs[2]:.1%} harmonic={fracs[3]:.1%}")

plt.subplots_adjust(wspace=0.08, hspace=0.16, left=0.12, right=0.98, top=0.94, bottom=0.03)
fig.savefig(FIG_DIR / "fig_dg_velocity_quiver.pdf", bbox_inches='tight')
fig.savefig(FIG_DIR / "fig_dg_velocity_quiver.png", bbox_inches='tight')
if "display" in globals():
    display(fig)
plt.close(fig)
print(f"Saved: {FIG_DIR / 'fig_dg_velocity_quiver.pdf'}")

gc.collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## QA

# COMMAND ----------

# DBTITLE 1,saved files
qa_path = OUT_DIR / "velocity_hodge.json"
print("json exists", qa_path.exists())
print("hodge figure exists", (FIG_DIR / "fig_dg_velocity_hodge.pdf").exists())
print("quiver figure exists", (FIG_DIR / "fig_dg_velocity_quiver.pdf").exists())
with open(qa_path) as f:
    qa_payload = json.load(f)

# COMMAND ----------

# DBTITLE 1,hodge pair count
print("hodge pairs", len(qa_payload["pairs"]), "expected", len(EPOCHS) - 1)

# COMMAND ----------

# DBTITLE 1,mean fractions
print("mean exact", qa_payload["mean_exact"])
print("mean coexact", qa_payload["mean_coexact"])
print("mean harmonic", qa_payload["mean_harmonic"])

# COMMAND ----------

# DBTITLE 1,fraction sums
qa_sums = [row["exact"] + row["coexact"] + row["harmonic"] for row in qa_payload["pairs"]]
print("first sums", qa_sums[:5])
print("sum range", min(qa_sums), max(qa_sums))

# COMMAND ----------

# DBTITLE 1,transition window values
for row in qa_payload["pairs"]:
    midpoint = row.get("midpoint", row.get("mid_epoch"))
    if 1500 <= midpoint <= 4000:
        print(row["epoch_a"], row["epoch_b"], row["exact"], row["coexact"], row["harmonic"])

# COMMAND ----------

# DBTITLE 1,test hodge decomp
qa_grid = np.linspace(-1.0, 1.0, 5)
qa_points = np.array([[x, y, 0.0] for x in qa_grid for y in qa_grid])
qa_velocity = np.tile(np.array([1.0, 0.0, 0.0]), (qa_points.shape[0], 1))
qa_hodge = hodge_decompose_velocity(qa_points, qa_velocity, PCA, dg, pca_dim=2, knn=5, n_basis=10)
print("known flow exact", qa_hodge["exact"])
print("known flow coexact", qa_hodge["coexact"])
print("known flow harmonic", qa_hodge["harmonic"])
