import csv
import json
import os
import string
from pathlib import Path
from typing import Any

import numpy as np

from grokking_summary_helpers import effective_dimension, mean_sd, summarise_transition_series
from runtime import ensure_dir, write_json


def _format_config_value(value: str, context: dict[str, str]) -> str:
    value = os.path.expandvars(value)
    missing = [field for _, field, _, _ in string.Formatter().parse(value) if field and field not in context]
    assert not missing, f"Unknown seed-sweep config placeholder {{{missing[0]}}}"
    return value.format(**context)


def _expand_config_values(obj: Any, context: dict[str, str]) -> Any:
    if isinstance(obj, str):
        return _format_config_value(obj, context)
    if isinstance(obj, list):
        return [_expand_config_values(item, context) for item in obj]
    if isinstance(obj, dict):
        return {key: _expand_config_values(value, context) for key, value in obj.items()}
    return obj


def load_seed_sweep_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    roots = {
        key: _format_config_value(str(value), {})
        for key, value in config.get("roots", {}).items()
    }
    config = _expand_config_values(config, roots)
    config["roots"] = roots
    return config


def expected_activation_epochs(training_defaults: dict[str, Any]) -> list[int]:
    n_epochs = int(training_defaults.get("n_epochs", 25000))
    save_every = int(training_defaults.get("save_every", 500))
    return list(range(0, n_epochs + 1, save_every))


def activation_status(activation_dir: str | Path, epochs: list[int]) -> dict[str, Any]:
    activation_dir = Path(activation_dir)
    required = [activation_dir / f"act_{epoch}.npy" for epoch in epochs]
    required.extend([activation_dir / "gt_labels.npy", activation_dir / "training.json"])
    missing = [str(path) for path in required if not path.exists()]
    existing_acts = list(activation_dir.glob("act_*.npy")) if activation_dir.exists() else []
    return {
        "complete": len(missing) == 0,
        "existing_activation_count": len(existing_acts),
        "missing": missing,
    }


# extra seeds write outside the canonical output root
def training_parameters(run: dict[str, Any], training_defaults: dict[str, Any]) -> dict[str, str]:
    params = {
        "THESIS_DATA_ROOT": str(run["output_root"]),
        "THESIS_SHARED_DIR": os.environ["THESIS_SHARED_DIR"],
        "GROKKING_ACTIVATION_DIR": str(run["activation_dir"]),
        "GROKKING_DATA_SEED": str(run["data_seed"]),
        "GROKKING_PROBE_SEED": str(run["probe_seed"]),
        "GROKKING_TMP_DIR": str(run.get("tmp_dir", f"/tmp/grokking_seed_sweep_{run['key']}")),
        "GROKKING_FORCE": str(bool(run.get("force_retrain", False))).lower(),
    }
    mapping = {
        "p": "GROKKING_P",
        "d_model": "GROKKING_D_MODEL",
        "n_epochs": "GROKKING_N_EPOCHS",
        "save_every": "GROKKING_SAVE_EVERY",
        "n_sub": "GROKKING_N_SUB",
        "train_frac": "GROKKING_TRAIN_FRAC",
    }
    for key, widget_name in mapping.items():
        if key in training_defaults:
            params[widget_name] = str(training_defaults[key])
    return params


def analysis_parameters(
    run: dict[str, Any],
    workspace_root: str,
    chart_variant_prefix: str,
    task_keys: list[str],
    verify_outputs: bool,
) -> dict[str, str]:
    return {
        "THESIS_WORKSPACE_ROOT": workspace_root,
        "THESIS_DATA_ROOT": str(run["output_root"]),
        "THESIS_SHARED_DIR": os.environ["THESIS_SHARED_DIR"],
        "GROKKING_ACTIVATION_DIR": str(run["activation_dir"]),
        "GROKKING_CHART_VARIANT": f"{chart_variant_prefix}_{run['data_seed']}",
        "TASKS": ",".join(task_keys),
        "VERIFY_OUTPUTS": str(verify_outputs).lower(),
    }


def first_epoch_at_threshold(saved_epochs: list[int], values: list[float], threshold: float) -> int | None:
    for epoch, value in zip(saved_epochs, values):
        if float(value) >= threshold:
            return int(epoch)
    return None


def read_json_if_present(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarise_effective_dimension(activation_dir: str | Path, final_epoch: int) -> dict[str, float | None]:
    activation_dir = Path(activation_dir)
    out: dict[str, float | None] = {}
    for epoch in [0, 3000, 4000, final_epoch]:
        path = activation_dir / f"act_{epoch}.npy"
        out[f"d_eff_{epoch}"] = effective_dimension(np.load(path).astype(np.float64)) if path.exists() else None
    d0 = out.get("d_eff_0")
    df = out.get(f"d_eff_{final_epoch}")
    out["drop_0_to_final"] = float(d0 - df) if d0 is not None and df is not None else None
    out["ratio_final_over_0"] = float(df / d0) if d0 not in (None, 0.0) and df is not None else None
    return out


def summarise_training(activation_dir: str | Path) -> dict[str, Any]:
    meta = read_json_if_present(Path(activation_dir) / "training.json")
    if meta is None:
        return {"available": False}
    saved_epochs = [int(x) for x in meta.get("saved_epochs", [])]
    val_accs = [float(x) for x in meta.get("val_accs", [])]
    return {
        "available": True,
        "config": meta.get("config", {}),
        "final_val_acc": float(val_accs[-1]) if val_accs else None,
        "first_epoch_val_ge_0_1": first_epoch_at_threshold(saved_epochs, val_accs, 0.1),
        "first_epoch_val_ge_0_5": first_epoch_at_threshold(saved_epochs, val_accs, 0.5),
        "first_epoch_val_ge_0_9": first_epoch_at_threshold(saved_epochs, val_accs, 0.9),
        "first_epoch_val_ge_0_99": first_epoch_at_threshold(saved_epochs, val_accs, 0.99),
    }


def summarise_bw_geodesic(output_root: str | Path) -> dict[str, Any]:
    data = read_json_if_present(Path(output_root) / "results" / "grokking_bw_geodesic" / "bw_geodesic_results.json")
    if data is None:
        return {"available": False}
    series = data.get("bw_distances_consecutive", {})
    return {
        "available": True,
        "summary": summarise_transition_series(series.get("midpoint_epochs", []), series.get("distances", [])),
        "key_pairs": data.get("key_pairs", {}),
    }


def summarise_heat_kernel(output_root: str | Path, tau: float = 1.0) -> dict[str, Any]:
    data = read_json_if_present(Path(output_root) / "results" / "grokking_heat_kernel" / "heat_kernel_bw_results.json")
    if data is None:
        return {"available": False}
    key = str(tau)
    series = data.get("bw_series", {}).get(key)
    if series is None:
        return {"available": False, "tau": tau}
    return {
        "available": True,
        "tau": tau,
        "summary": summarise_transition_series(series.get("midpoint_epochs", []), series.get("distances", [])),
    }


def summarise_spectral_perturbation(output_root: str | Path) -> dict[str, Any]:
    data = read_json_if_present(
        Path(output_root) / "results" / "grokking_spectral_perturbation" / "spectral_perturbation_results.json"
    )
    if data is None:
        return {"available": False}
    return {"available": True, "summary": data.get("summary", {})}


def summarise_subspace_bw(output_root: str | Path) -> dict[str, Any]:
    data = read_json_if_present(Path(output_root) / "results" / "grokking_subspace_bw" / "subspace_bw_results.json")
    if data is None:
        return {"available": False}
    midpoints = []
    for label in data.get("epochs", []):
        left, right = [float(x) for x in str(label).split("-", 1)]
        midpoints.append(0.5 * (left + right))
    return {
        "available": True,
        "full": summarise_transition_series(midpoints, data.get("full", [])),
    }


def summarise_seed_run(run: dict[str, Any], training_defaults: dict[str, Any]) -> dict[str, Any]:
    final_epoch = int(training_defaults.get("n_epochs", 25000))
    activation_dir = Path(run["activation_dir"])
    output_root = Path(run["output_root"])
    return {
        "key": run["key"],
        "label": run["label"],
        "canonical": bool(run.get("canonical", False)),
        "data_seed": int(run["data_seed"]),
        "probe_seed": int(run["probe_seed"]),
        "root": str(run["root"]),
        "activation_dir": str(activation_dir),
        "output_root": str(output_root),
        "training": summarise_training(activation_dir),
        "effective_dimension": summarise_effective_dimension(activation_dir, final_epoch),
        "bw_geodesic": summarise_bw_geodesic(output_root),
        "heat_kernel_tau_1": summarise_heat_kernel(output_root, tau=1.0),
        "spectral_perturbation": summarise_spectral_perturbation(output_root),
        "subspace_bw": summarise_subspace_bw(output_root),
    }


def nested_value(obj: dict[str, Any], path: list[str]) -> Any:
    cur: Any = obj
    for item in path:
        if not isinstance(cur, dict) or item not in cur:
            return None
        cur = cur[item]
    return cur


def aggregate_seed_summaries(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "effective_dimension_ratio_final_over_0": ["effective_dimension", "ratio_final_over_0"],
        "bw_transition_peak_over_post": ["bw_geodesic", "summary", "transition_peak_over_post_mean"],
        "bw_transition_mean_over_post": ["bw_geodesic", "summary", "transition_mean_over_post_mean"],
        "heat_tau_1_transition_peak_over_post": ["heat_kernel_tau_1", "summary", "transition_peak_over_post_mean"],
        "spectral_transition_drift": ["spectral_perturbation", "summary", "transition_drift"],
        "spectral_transition_mixing": ["spectral_perturbation", "summary", "transition_mixing"],
        "subspace_full_transition_peak_over_post": ["subspace_bw", "full", "transition_peak_over_post_mean"],
    }
    return {
        name: mean_sd([nested_value(summary, path) for summary in run_summaries], ignore_none=True, include_n=True)
        for name, path in metrics.items()
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def seed_summary_rows(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in run_summaries:
        rows.append(
            {
                "label": summary["label"],
                "data_seed": summary["data_seed"],
                "probe_seed": summary["probe_seed"],
                "val_ge_0_5_epoch": nested_value(summary, ["training", "first_epoch_val_ge_0_5"]),
                "val_ge_0_99_epoch": nested_value(summary, ["training", "first_epoch_val_ge_0_99"]),
                "d_eff_final_over_0": nested_value(summary, ["effective_dimension", "ratio_final_over_0"]),
                "bw_peak_over_post": nested_value(summary, ["bw_geodesic", "summary", "transition_peak_over_post_mean"]),
                "heat_tau_1_peak_over_post": nested_value(
                    summary, ["heat_kernel_tau_1", "summary", "transition_peak_over_post_mean"]
                ),
                "spectral_transition_drift": nested_value(
                    summary, ["spectral_perturbation", "summary", "transition_drift"]
                ),
                "spectral_transition_mixing": nested_value(
                    summary, ["spectral_perturbation", "summary", "transition_mixing"]
                ),
                "subspace_peak_over_post": nested_value(
                    summary, ["subspace_bw", "full", "transition_peak_over_post_mean"]
                ),
            }
        )
    return rows


def write_summary_table(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    headers = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary_markdown(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    headers = list(rows[0].keys()) if rows else []
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header)) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_seed_sweep_outputs(summary_dir: str | Path, payload: dict[str, Any]) -> dict[str, str]:
    summary_dir = ensure_dir(summary_dir)
    rows = seed_summary_rows(payload["run_summaries"])
    json_path = write_json(summary_dir / "seed_sweep_summary.json", payload)
    csv_path = write_summary_table(summary_dir / "seed_sweep_summary.csv", rows)
    md_path = write_summary_markdown(summary_dir / "seed_sweep_summary.md", rows)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
