# Databricks notebook source

# MAGIC %md
# MAGIC # Grokking Pipeline: Run Analysis Notebooks
# MAGIC
# MAGIC Runs the selected Grokking analysis tasks and verifies their JSON and figure
# MAGIC outputs and writes figures under the configured thesis data root.

# COMMAND ----------


from pathlib import Path

# COMMAND ----------

import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["THESIS_SHARED_DIR"])
from pipeline_helpers import *


WORKSPACE_ROOT = workspace_root()
REPRO_ROOT = reproduction_code_dir(WORKSPACE_ROOT)
ANALYSIS_ROOT = f"{REPRO_ROOT}/Grokking/Analysis"
DATA_ROOT = thesis_data_root()
DBFS_FIGURE_ROOT = DATA_ROOT / "figures"
DEFAULT_TIMEOUT_SECONDS = 0


CHART_VARIANT = widget_or_default("GROKKING_CHART_VARIANT", "30_04_2026_new_charts")
PHASE_CMAP = widget_or_default("GROKKING_PHASE_CMAP", "hsv")
TASK_FILTER = widget_or_default("TASKS", "").strip()
DRY_RUN = bool_widget("DRY_RUN", False)
VERIFY_OUTPUTS = bool_widget("VERIFY_OUTPUTS", True)

FIG_SUBDIR = "grokking" if CHART_VARIANT in {"", "original", "legacy"} else f"grokking_{CHART_VARIANT}"
RUN_PARAMS = {
    "THESIS_DATA_ROOT": str(thesis_data_root()),
    "THESIS_WORKSPACE_ROOT": WORKSPACE_ROOT,
    "THESIS_SHARED_DIR": os.environ["THESIS_SHARED_DIR"],
    "GROKKING_CHART_VARIANT": CHART_VARIANT,
    "GROKKING_PHASE_CMAP": PHASE_CMAP,
}
for param_name in [
    "GROKKING_ACTIVATION_DIR",
    "GROKKING_FIGURE_DIR",
    "GROKKING_PHASE_SATURATION",
    "GROKKING_PHASE_VALUE",
]:
    param_value = widget_or_default(param_name, "").strip()
    if param_value:
        RUN_PARAMS[param_name] = param_value


TASKS: tuple[PipelineTask, ...] = (
    PipelineTask(
        "velocity_hodge",
        "01_velocity_hodge",
        ("fig_dg_velocity_hodge.png", "fig_dg_velocity_hodge.pdf", "fig_dg_velocity_quiver.png", "fig_dg_velocity_quiver.pdf"),
        ANALYSIS_ROOT,
        (str(DATA_ROOT / "results" / "grokking_dg_velocity_hodge" / "velocity_hodge.json"),),
    ),
    PipelineTask(
        "effdim_pca",
        "02_effdim_pca",
        (
            "fig_effective_dim.png",
            "fig_effective_dim.pdf",
            "fig_fourier_ridge_pca.png",
            "fig_fourier_ridge_pca.pdf",
            "fig_pca_grid.png",
            "fig_pca_grid.pdf",
        ),
        ANALYSIS_ROOT,
    ),
    PipelineTask("eigenvalues", "03_eigenvalues", ("fig_eigenvalue_evolution.png", "fig_eigenvalue_evolution.pdf"), ANALYSIS_ROOT),
    PipelineTask("heat_kernel", "05_heat_kernel", ("fig_heat_kernel_bw.png", "fig_heat_kernel_bw.pdf"), ANALYSIS_ROOT, (str(DATA_ROOT / "results" / "grokking_heat_kernel" / "heat_kernel_bw_results.json"),)),
    PipelineTask("circular_coords", "07_circular_coords", ("fig_circular_coords.png", "fig_circular_coords.pdf"), ANALYSIS_ROOT, (str(DATA_ROOT / "results" / "grokking_circular_coords" / "circular_results.json"),)),
    PipelineTask("probe_subset_robustness", "09_probe_subset_robustness", (), ANALYSIS_ROOT, (str(DATA_ROOT / "results" / "grokking_probe_subset_robustness_28_04_2026" / "09_probe_subset_robustness.json"),)),
    PipelineTask(
        "event_study",
        "10_event_study",
        ("fig_grokking_event_study.png", "fig_grokking_event_study.pdf"),
        ANALYSIS_ROOT,
        (str(DATA_ROOT / "results" / "grokking_event_study" / "event_study_results.json"),),
    ),
)

run_notebook_pipeline(
    "Grokking",
    TASKS,
    TASK_FILTER,
    RUN_PARAMS,
    DBFS_FIGURE_ROOT,
    FIG_SUBDIR,
    DATA_ROOT,
    timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
    dry_run=DRY_RUN,
    verify_outputs=VERIFY_OUTPUTS,
)
