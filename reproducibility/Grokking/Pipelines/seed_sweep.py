"""Train and analyse the configured multi-seed robustness sweep."""

import argparse
from pathlib import Path

from grokking_velocity_hodge.pipeline import (
    bool_widget,
    reproduction_code_dir,
    run_workspace_notebook,
    widget_or_default,
    workspace_root,
)
from grokking_velocity_hodge.seed_sweep import (
    activation_status,
    aggregate_seed_summaries,
    analysis_parameters,
    expected_activation_epochs,
    load_seed_sweep_config,
    summarise_seed_run,
    training_parameters,
    write_seed_sweep_outputs,
)


def resolve_config_path(spec: str, workspace_root_value: str) -> Path:
    spec = spec.strip()
    repro_root = Path(reproduction_code_dir(workspace_root_value))
    if not spec:
        local_default = repro_root / "Grokking" / "config" / "seed_sweep.json"
        if local_default.exists():
            return local_default
        return Path(
            "/Workspace"
            + f"{workspace_root_value}/Thesis_Reproducibility_Code/Grokking/config/seed_sweep.json"
        )
    path = Path(spec)
    if path.exists():
        return path
    if spec.startswith("/Users/"):
        workspace_path = Path("/Workspace" + spec)
        if workspace_path.exists():
            return workspace_path
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to the seed-sweep JSON configuration.")
    parser.add_argument("--seeds", help="Comma-separated run keys or data seeds.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without running it.")
    parser.add_argument("--no-training", action="store_true", help="Do not train missing runs.")
    parser.add_argument("--no-analysis", action="store_true", help="Do not run analyses.")
    parser.add_argument("--no-verify", action="store_true", help="Do not verify analysis outputs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    WORKSPACE_ROOT = workspace_root()
    REPRO_ROOT = reproduction_code_dir(WORKSPACE_ROOT)
    config_spec = args.config if args.config is not None else widget_or_default("SEED_SWEEP_CONFIG", "")
    CONFIG_PATH = resolve_config_path(config_spec, WORKSPACE_ROOT)
    CONFIG = load_seed_sweep_config(CONFIG_PATH)

    DRY_RUN = args.dry_run or bool_widget("DRY_RUN", False)
    RUN_TRAINING = False if args.no_training else bool_widget("RUN_TRAINING", True)
    RUN_ANALYSIS = False if args.no_analysis else bool_widget("RUN_ANALYSIS", True)
    VERIFY_OUTPUTS = False if args.no_verify else bool_widget("VERIFY_OUTPUTS", True)
    SEED_FILTER = args.seeds if args.seeds is not None else widget_or_default("SEEDS", "").strip()

    TRAINING_ROOT = f"{REPRO_ROOT}/Grokking/Training"
    PIPELINE_ROOT = f"{REPRO_ROOT}/Grokking/Pipelines"
    TRAINING_NOTEBOOK = f"{TRAINING_ROOT}/01_train_reproducible"
    MAIN_CHARTS_NOTEBOOK = f"{PIPELINE_ROOT}/main_charts"

    training_defaults = CONFIG.get("training_defaults", {})
    epochs = expected_activation_epochs(training_defaults)
    analysis_tasks = list(CONFIG.get("analysis_tasks", []))
    chart_variant_prefix = CONFIG.get("chart_variant_prefix", "seed_sweep")

    runs = list(CONFIG.get("runs", []))
    if SEED_FILTER:
        requested = {item.strip() for item in SEED_FILTER.split(",") if item.strip()}
        runs = [run for run in runs if run["key"] in requested or str(run["data_seed"]) in requested]
        missing = requested.difference({run["key"] for run in runs}).difference(
            {str(run["data_seed"]) for run in runs}
        )
        assert not missing, f"Unknown seed-sweep run(s): {sorted(missing)}"

    print("Grokking seed-sweep pipeline")
    print(f"Config: {CONFIG_PATH}")
    print(f"Runs: {[run['key'] for run in runs]}")
    print(f"Analysis tasks: {analysis_tasks}")
    print(f"Dry run: {DRY_RUN}; run training: {RUN_TRAINING}; run analysis: {RUN_ANALYSIS}")

    for run in runs:
        print("=" * 88)
        print(f"Seed run: {run['key']} ({run['label']})")
        status = activation_status(run["activation_dir"], epochs)
        print(f"Activation dir: {run['activation_dir']}")
        print(
            f"Activation status: complete={status['complete']}, "
            f"existing={status['existing_activation_count']}"
        )

        if status["complete"]:
            print("Training skipped: activation set is already complete.")
            continue

        if DRY_RUN:
            print("Training would run because activations are incomplete.")
            continue

        assert RUN_TRAINING, f"Activation set is incomplete and RUN_TRAINING=false: {status['missing'][:5]}"

        assert not run.get("canonical", False), "Canonical activations are missing"

        assert not (status["existing_activation_count"] and not run.get("force_retrain", False)), (
            f"{run['activation_dir']} contains a partial activation set"
        )

        params = training_parameters(run, training_defaults)
        run_workspace_notebook(TRAINING_NOTEBOOK, params, timeout_seconds=0, dry_run=False)

        status = activation_status(run["activation_dir"], epochs)
        assert status["complete"], (
            f"Training finished but activation set is still incomplete: {status['missing'][:5]}"
        )

    for run in runs:
        print("=" * 88)
        print(f"Analysis run: {run['key']} ({run['label']})")
        params = analysis_parameters(
            run, WORKSPACE_ROOT, chart_variant_prefix, analysis_tasks, VERIFY_OUTPUTS
        )
        run_workspace_notebook(
            MAIN_CHARTS_NOTEBOOK, params, timeout_seconds=0, dry_run=DRY_RUN or not RUN_ANALYSIS
        )

    if DRY_RUN:
        print("Dry run complete; summary files were not written.")
    else:
        run_summaries = [summarise_seed_run(run, training_defaults) for run in runs]
        payload = {
            "config_path": str(CONFIG_PATH),
            "analysis_tasks": analysis_tasks,
            "training_defaults": training_defaults,
            "run_summaries": run_summaries,
            "aggregate": aggregate_seed_summaries(run_summaries),
        }
        written = write_seed_sweep_outputs(CONFIG["summary_dir"], payload)
        print("=" * 88)
        print("Seed-sweep summary files:")
        for kind, path in written.items():
            print(f"{kind}: {path}")

    print("=" * 88)
    print("Grokking seed sweep complete.")


if __name__ == "__main__":
    main()
