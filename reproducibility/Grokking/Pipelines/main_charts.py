"""Run the publication-figure analysis pipeline."""

import argparse

from grokking_velocity_hodge.pipeline import (
    PipelineTask,
    bool_widget,
    reproduction_code_dir,
    run_notebook_pipeline,
    thesis_data_root,
    widget_or_default,
    workspace_root,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print planned tasks without running them.")
    parser.add_argument("--no-verify", action="store_true", help="Do not verify expected output files.")
    parser.add_argument("--tasks", help="Comma-separated task names to run.")
    parser.add_argument("--chart-variant", help="Figure-output variant name.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    WORKSPACE_ROOT = workspace_root()
    REPRO_ROOT = reproduction_code_dir(WORKSPACE_ROOT)
    ANALYSIS_ROOT = f"{REPRO_ROOT}/Grokking/Analysis"
    DATA_ROOT = thesis_data_root()
    DBFS_FIGURE_ROOT = DATA_ROOT / "figures"
    DEFAULT_TIMEOUT_SECONDS = 0

    CHART_VARIANT = args.chart_variant or widget_or_default("GROKKING_CHART_VARIANT", "paper")
    PHASE_CMAP = widget_or_default("GROKKING_PHASE_CMAP", "hsv")
    TASK_FILTER = args.tasks if args.tasks is not None else widget_or_default("TASKS", "").strip()
    DRY_RUN = args.dry_run or bool_widget("DRY_RUN", False)
    VERIFY_OUTPUTS = False if args.no_verify else bool_widget("VERIFY_OUTPUTS", True)

    FIG_SUBDIR = "grokking" if CHART_VARIANT in {"", "original", "legacy"} else f"grokking_{CHART_VARIANT}"
    RUN_PARAMS = {
        "THESIS_DATA_ROOT": str(DATA_ROOT),
        "THESIS_WORKSPACE_ROOT": WORKSPACE_ROOT,
        "GROKKING_CHART_VARIANT": CHART_VARIANT,
        "GROKKING_PHASE_CMAP": PHASE_CMAP,
    }
    for param_name in [
        "GROKKING_ACTIVATION_DIR",
        "GROKKING_FIGURE_DIR",
        "GROKKING_PHASE_SATURATION",
        "GROKKING_PHASE_VALUE",
        "GROKKING_PCA_DIM",
        "GROKKING_PCA_SOLVER",
        "GROKKING_KNN",
        "GROKKING_K_SPEC",
        "GROKKING_RESOLVENT_EPS",
        "GROKKING_HODGE_PCA_DIM",
        "GROKKING_HODGE_BASIS",
        "GROKKING_HODGE_QUIVER_POINTS",
        "GROKKING_CIRCULAR_PCA_DIM",
        "GROKKING_CIRCULAR_BASIS",
        "GROKKING_CIRCULAR_EIGENPAIRS",
        "GROKKING_HEAT_SCALES",
        "GROKKING_PROBE_SUBSET_SIZE",
        "GROKKING_PROBE_SUBSET_SEEDS",
        "GROKKING_TRANSITION_START",
        "GROKKING_TRANSITION_END",
        "GROKKING_EVENT_TRANSITION_DELTA",
        "GROKKING_EVENT_X_LIMITS",
        "GROKKING_EVENT_GRID_STEP",
        "GROKKING_EVENT_ALIGNMENT_THRESHOLD",
    ]:
        param_value = widget_or_default(param_name, "").strip()
        if param_value:
            RUN_PARAMS[param_name] = param_value

    TASKS: tuple[PipelineTask, ...] = (
        PipelineTask(
            "velocity_hodge",
            "01_velocity_hodge",
            (
                "fig_dg_velocity_hodge.png",
                "fig_dg_velocity_hodge.pdf",
                "fig_dg_velocity_quiver.png",
                "fig_dg_velocity_quiver.pdf",
            ),
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
        PipelineTask(
            "eigenvalues",
            "03_eigenvalues",
            ("fig_eigenvalue_evolution.png", "fig_eigenvalue_evolution.pdf"),
            ANALYSIS_ROOT,
        ),
        PipelineTask(
            "resolvent_bw",
            "04_resolvent_bw",
            ("fig_resolvent_bw.png", "fig_resolvent_bw.pdf"),
            ANALYSIS_ROOT,
            (str(DATA_ROOT / "results" / "grokking_resolvent_bw" / "resolvent_bw_results.json"),),
        ),
        PipelineTask(
            "heat_kernel",
            "05_heat_kernel",
            ("fig_heat_kernel_bw.png", "fig_heat_kernel_bw.pdf"),
            ANALYSIS_ROOT,
            (str(DATA_ROOT / "results" / "grokking_heat_kernel" / "heat_kernel_bw_results.json"),),
        ),
        PipelineTask(
            "circular_coords",
            "07_circular_coords",
            ("fig_circular_coords.png", "fig_circular_coords.pdf"),
            ANALYSIS_ROOT,
            (str(DATA_ROOT / "results" / "grokking_circular_coords" / "circular_results.json"),),
        ),
        PipelineTask(
            "probe_subset_robustness",
            "09_probe_subset_robustness",
            (),
            ANALYSIS_ROOT,
            (
                str(
                    DATA_ROOT
                    / "results"
                    / "grokking_probe_subset_robustness"
                    / "09_probe_subset_robustness.json"
                ),
            ),
        ),
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


if __name__ == "__main__":
    main()
