# Databricks notebook source

# MAGIC %md
# MAGIC # Grokking 09: Probe Subset Robustness
# MAGIC
# MAGIC Inputs:
# MAGIC - Checkpoint activation snapshots
# MAGIC
# MAGIC Outputs:
# MAGIC - Subset robustness summaries for effective dimension and BW diagnostics
# MAGIC - JSON: 09_probe_subset_robustness.json

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ["THESIS_SHARED_DIR"])
from grokking_control_helpers import aggregate_probe_subsets, analyse_probe_subset
from runtime import configure_grokking_runtime, write_json

GROKKING = configure_grokking_runtime()
ACT_DIR = GROKKING.activation_dir
OUT_DIR = GROKKING.result_dir("grokking_probe_subset_robustness_28_04_2026")

EPOCHS = list(range(0, 25001, 500))
SUBSET_SIZE = 400
SUBSET_SEEDS = [1101, 2202, 3303, 4404]
PCA_DIM = 20
KNN = 15
K_SPEC = 30
EPS = 0.01
TAU_VALUES = [0.1, 1.0, 10.0]

print(f"ACT_DIR={ACT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Draw probe subsets

# COMMAND ----------

x0 = np.load(ACT_DIR / "act_0.npy")
n_total = x0.shape[0]
del x0
print(f"n_total={n_total}, subset_size={SUBSET_SIZE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Recompute diagnostics on each subset

# COMMAND ----------

subsets = []
for seed in SUBSET_SEEDS:
    rng = np.random.default_rng(seed)
    subset_idx = np.sort(rng.choice(n_total, size=SUBSET_SIZE, replace=False))
    subsets.append(
        analyse_probe_subset(
            ACT_DIR,
            seed,
            subset_idx,
            epochs=EPOCHS,
            tau_values=TAU_VALUES,
            pca_dim=PCA_DIM,
            knn=KNN,
            k_spec=K_SPEC,
            eps=EPS,
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Aggregate subset robustness

# COMMAND ----------

aggregate = aggregate_probe_subsets(subsets, TAU_VALUES)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Output: 09_probe_subset_robustness.json

# COMMAND ----------

# DBTITLE 1,09_probe_subset_robustness.json
payload = {
    "config": {
        "subset_size": SUBSET_SIZE,
        "subset_seeds": SUBSET_SEEDS,
        "epochs": EPOCHS,
        "pca_dim": PCA_DIM,
        "knn": KNN,
        "k_spec": K_SPEC,
        "eps": EPS,
        "tau_values": TAU_VALUES,
        "activation_dir": str(ACT_DIR),
    },
    "subsets": subsets,
    "aggregate": aggregate,
}

write_json(OUT_DIR / "09_probe_subset_robustness.json", payload)
write_json(OUT_DIR / "probe_subset_robustness.json", payload)

print(f"Aggregate: {aggregate}")
print("Saved 09 probe-subset robustness.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## QA

# COMMAND ----------

# DBTITLE 1,saved files
qa_out_dir = Path(OUT_DIR)
print("json exists", (qa_out_dir / "09_probe_subset_robustness.json").exists())

# COMMAND ----------

# DBTITLE 1,subset count
print("subsets", len(subsets), "expected", len(SUBSET_SEEDS))
print("subset seeds", [row["seed"] for row in subsets])

# COMMAND ----------

# DBTITLE 1,heat tau keys
print("expected tau", TAU_VALUES)
print("heat tau keys", sorted(aggregate["heat_kernel"].keys()))

# COMMAND ----------

# DBTITLE 1,collapse ratio
print("collapse ratio mean", aggregate["effective_dimension"]["ratio_25000_over_0"]["mean"])
collapse_ratio = aggregate["effective_dimension"]["ratio_25000_over_0"]
print("collapse ratio sd", collapse_ratio.get("sd", collapse_ratio.get("std")))

# COMMAND ----------

# DBTITLE 1,subset overlap sizes
for i in range(len(subsets)):
    for j in range(i + 1, len(subsets)):
        a = set(subsets[i]["subset_idx"])
        b = set(subsets[j]["subset_idx"])
        print(subsets[i]["seed"], subsets[j]["seed"], "overlap", len(a & b))
