"""Cross-seed reporting for the empirical Hodge robustness experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import t

from .runtime import ensure_dir, write_json

COMPONENTS = ("exact", "coexact", "harmonic")
BASELINE_SETTING = (10, 15, 50)


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values:
        raise ValueError(f"Cannot calculate {key}: the selected group is empty")
    return float(np.mean(values))


def _t_interval(values: Iterable[float], confidence: float = 0.95) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) < 2:
        raise ValueError("A seed-level t interval requires at least two training seeds")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    quantile = float(t.ppf((1.0 + confidence) / 2.0, df=len(array) - 1))
    half_width = quantile * sd / math.sqrt(len(array))
    return {
        "n_training_seeds": int(len(array)),
        "mean": mean,
        "sd": sd,
        "confidence": confidence,
        "method": "two-sided Student t interval over training-seed means",
        "interval": [mean - half_width, mean + half_width],
    }


def _first_epoch_at_threshold(training: dict[str, Any], threshold: float) -> int:
    epochs = training.get("saved_epochs", [])
    values = training.get("val_accs", training.get("test_accs", []))
    for epoch, value in zip(epochs, values):
        if float(value) >= threshold:
            return int(epoch)
    raise ValueError(f"Validation accuracy never reaches {threshold}")


def phase_bounds(
    training: dict[str, Any],
    convention: str,
    fixed_window: tuple[float, float] = (1500.0, 4000.0),
    alignment_threshold: float = 0.5,
    alignment_offsets: tuple[float, float] = (-1500.0, 1000.0),
) -> dict[str, float | int | str | None]:
    if convention == "fixed":
        return {
            "convention": convention,
            "alignment_epoch": None,
            "transition_start": float(fixed_window[0]),
            "transition_end": float(fixed_window[1]),
        }
    if convention != "event_aligned":
        raise ValueError(f"Unknown phase convention: {convention}")
    alignment_epoch = _first_epoch_at_threshold(training, alignment_threshold)
    return {
        "convention": convention,
        "alignment_epoch": alignment_epoch,
        "transition_start": float(alignment_epoch + alignment_offsets[0]),
        "transition_end": float(alignment_epoch + alignment_offsets[1]),
    }


def classify_phase(midpoint: float, bounds: dict[str, Any]) -> str:
    if midpoint < float(bounds["transition_start"]):
        return "pre"
    if midpoint > float(bounds["transition_end"]):
        return "post"
    return "transition"


def _setting(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["pca_dim"]), int(row["knn"]), int(row["n_basis"])


def _setting_dict(setting: tuple[int, int, int]) -> dict[str, int]:
    return {"pca_dim": setting[0], "knn": setting[1], "n_basis": setting[2]}


def _record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["subset_seed"]),
        int(row["epoch_a"]),
        int(row["epoch_b"]),
        *_setting(row),
    )


def _validate_run(seed: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [(_record_key(row), bool(row["permuted_correspondence"])) for row in rows]
    errors = [
        abs(float(row["exact"]) + float(row["coexact"]) + float(row["harmonic"]) - 1.0)
        for row in rows
    ]
    ordinary = [row for row in rows if not bool(row["permuted_correspondence"])]
    null = [row for row in rows if bool(row["permuted_correspondence"])]
    baseline = [row for row in ordinary if _setting(row) == BASELINE_SETTING]
    return {
        "training_seed": seed,
        "records": len(rows),
        "ordinary_records": len(ordinary),
        "permutation_records": len(null),
        "parameter_settings": len({_setting(row) for row in ordinary}),
        "probe_subset_seeds": sorted({int(row["subset_seed"]) for row in rows}),
        "baseline_checkpoint_pairs": len(
            {(int(row["epoch_a"]), int(row["epoch_b"])) for row in baseline}
        ),
        "unique_records": len(set(keys)),
        "all_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (*COMPONENTS, "total_energy")
        ),
        "max_fraction_sum_error": max(errors, default=0.0),
    }


def _summarise_convention(
    runs: list[dict[str, Any]],
    convention: str,
    fixed_window: tuple[float, float],
    alignment_threshold: float,
    alignment_offsets: tuple[float, float],
) -> dict[str, Any]:
    baseline_seed_phase = []
    contrasts = []
    permutation_seed_phase = []
    phase_sampling = []
    settings_seen: set[tuple[int, int, int]] = set()

    for run in runs:
        seed = int(run["training_seed"])
        rows = run["records"]
        bounds = phase_bounds(
            run["training"],
            convention,
            fixed_window=fixed_window,
            alignment_threshold=alignment_threshold,
            alignment_offsets=alignment_offsets,
        )
        labelled = [dict(row, reporting_phase=classify_phase(float(row["midpoint"]), bounds)) for row in rows]
        ordinary = [row for row in labelled if not bool(row["permuted_correspondence"])]
        baseline_actual = [row for row in ordinary if _setting(row) == BASELINE_SETTING]
        baseline_null = [
            row
            for row in labelled
            if bool(row["permuted_correspondence"]) and _setting(row) == BASELINE_SETTING
        ]
        settings = sorted({_setting(row) for row in ordinary})
        settings_seen.update(settings)

        seed_phase: dict[str, Any] = {
            "training_seed": seed,
            "phase_bounds": bounds,
        }
        for phase in ("pre", "transition", "post"):
            phase_rows = [row for row in baseline_actual if row["reporting_phase"] == phase]
            component_means = {component: _mean(phase_rows, component) for component in COMPONENTS}
            component_means["coexact_minus_exact"] = (
                component_means["coexact"] - component_means["exact"]
            )
            seed_phase[phase] = component_means
            phase_sampling.append(
                {
                    "training_seed": seed,
                    "phase": phase,
                    "checkpoint_pairs": len(
                        {(int(row["epoch_a"]), int(row["epoch_b"])) for row in phase_rows}
                    ),
                    "ordinary_records": len(phase_rows),
                }
            )
        seed_phase["post_minus_transition_coexact_minus_exact"] = (
            seed_phase["post"]["coexact_minus_exact"]
            - seed_phase["transition"]["coexact_minus_exact"]
        )
        baseline_seed_phase.append(seed_phase)

        for setting in settings:
            setting_rows = [row for row in ordinary if _setting(row) == setting]
            transition_rows = [
                row for row in setting_rows if row["reporting_phase"] == "transition"
            ]
            post_rows = [row for row in setting_rows if row["reporting_phase"] == "post"]
            transition_difference = _mean(transition_rows, "coexact") - _mean(
                transition_rows, "exact"
            )
            post_difference = _mean(post_rows, "coexact") - _mean(post_rows, "exact")
            contrasts.append(
                {
                    "training_seed": seed,
                    **_setting_dict(setting),
                    "transition_coexact_minus_exact": transition_difference,
                    "post_coexact_minus_exact": post_difference,
                    "post_minus_transition": post_difference - transition_difference,
                }
            )

        actual_by_key = {_record_key(row): row for row in baseline_actual}
        null_by_key = {_record_key(row): row for row in baseline_null}
        if set(actual_by_key) != set(null_by_key):
            raise ValueError(f"Seed {seed}: actual and permutation-null baseline records do not pair")
        for phase in ("pre", "transition", "post"):
            keys = [
                key
                for key, row in actual_by_key.items()
                if row["reporting_phase"] == phase
            ]
            permutation_seed_phase.append(
                {
                    "training_seed": seed,
                    "phase": phase,
                    "paired_permutations": len(keys),
                    **{
                        f"actual_minus_null_{component}": float(
                            np.mean(
                                [
                                    float(actual_by_key[key][component])
                                    - float(null_by_key[key][component])
                                    for key in keys
                                ]
                            )
                        )
                        for component in COMPONENTS
                    },
                }
            )

    baseline_phase_summary = {}
    for phase in ("pre", "transition", "post"):
        baseline_phase_summary[phase] = {
            component: _t_interval([row[phase][component] for row in baseline_seed_phase])
            for component in (*COMPONENTS, "coexact_minus_exact")
        }

    baseline_changes = [
        row["post_minus_transition_coexact_minus_exact"] for row in baseline_seed_phase
    ]
    post_permutation = [row for row in permutation_seed_phase if row["phase"] == "post"]
    inference = {
        "baseline_post_minus_transition_coexact_minus_exact": _t_interval(baseline_changes),
        "post_actual_minus_null_exact": _t_interval(
            [row["actual_minus_null_exact"] for row in post_permutation]
        ),
        "post_actual_minus_null_coexact": _t_interval(
            [row["actual_minus_null_coexact"] for row in post_permutation]
        ),
        "independent_unit": "training seed",
        "within_seed_aggregation": (
            "Checkpoint-pair and probe-subset values are averaged within each training seed before "
            "the across-seed interval is calculated."
        ),
    }
    direction_counts = {
        "seed_setting_shift_toward_exact": {
            "count": sum(row["post_minus_transition"] < 0.0 for row in contrasts),
            "total": len(contrasts),
            "independence_note": (
                "The seven settings within a seed are correlated sensitivity comparisons, not "
                "independent replications."
            ),
        },
        "seed_setting_post_exact_exceeds_coexact": {
            "count": sum(row["post_coexact_minus_exact"] < 0.0 for row in contrasts),
            "total": len(contrasts),
        },
    }

    return {
        "description": (
            "Fixed baseline epochs 1500--4000."
            if convention == "fixed"
            else (
                "Seed-relative window anchored at the first checkpoint with validation accuracy at "
                f"least {alignment_threshold}, using offsets {alignment_offsets}. The same nine "
                "preselected checkpoint pairs are reclassified, so the number of pairs per phase can vary."
            )
        ),
        "baseline_setting": _setting_dict(BASELINE_SETTING),
        "settings": [_setting_dict(setting) for setting in sorted(settings_seen)],
        "baseline_by_seed_phase": baseline_seed_phase,
        "baseline_phase_summary": baseline_phase_summary,
        "seed_setting_contrasts": contrasts,
        "direction_counts": direction_counts,
        "permutation_effect_by_seed": permutation_seed_phase,
        "phase_sampling": phase_sampling,
        "seed_level_inference": inference,
    }


def build_cross_seed_hodge_summary(
    run_inputs: Iterable[dict[str, Any]],
    fixed_window: tuple[float, float] = (1500.0, 4000.0),
    alignment_threshold: float = 0.5,
    alignment_offsets: tuple[float, float] = (-1500.0, 1000.0),
) -> dict[str, Any]:
    runs = []
    integrity = []
    source_files = []
    for spec in run_inputs:
        seed = int(spec["training_seed"])
        hodge_path = Path(spec["hodge_path"])
        training_path = Path(spec["training_path"])
        hodge = _read_json(hodge_path)
        training = _read_json(training_path)
        rows = list(hodge.get("records", []))
        integrity.append(_validate_run(seed, rows))
        runs.append({"training_seed": seed, "records": rows, "training": training})
        source_files.extend(
            [
                {
                    "path": hodge_path.name,
                    "sha256": _sha256(hodge_path),
                    "bytes": hodge_path.stat().st_size,
                },
                {
                    "path": training_path.name,
                    "sha256": _sha256(training_path),
                    "bytes": training_path.stat().st_size,
                },
            ]
        )

    runs.sort(key=lambda run: run["training_seed"])
    seeds = [run["training_seed"] for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Training seeds must be unique")
    if len(seeds) != 8:
        raise ValueError(f"The manuscript report requires exactly eight training seeds; found {len(seeds)}")
    expected_shape = {
        "records": 288,
        "ordinary_records": 252,
        "permutation_records": 36,
        "parameter_settings": 7,
        "probe_subset_seeds": [1101, 2202, 3303, 4404],
        "baseline_checkpoint_pairs": 9,
    }
    if any(
        not row["all_finite"]
        or row["records"] != row["unique_records"]
        or any(row[key] != value for key, value in expected_shape.items())
        or row["max_fraction_sum_error"] > 1e-8
        for row in integrity
    ):
        raise ValueError("Hodge artifact integrity validation failed")

    total_records = sum(row["records"] for row in integrity)
    ordinary_records = sum(row["ordinary_records"] for row in integrity)
    permutation_records = sum(row["permutation_records"] for row in integrity)
    return {
        "schema_version": 1,
        "design": {
            "training_seeds": seeds,
            "n_training_seeds": len(seeds),
            "independent_unit": "training seed",
            "probe_subset_seeds": [1101, 2202, 3303, 4404],
            "n_probe_subsets": 4,
            "n_parameter_settings": 7,
            "preselected_checkpoint_pairs": 9,
            "permutation_scheme": (
                "One separately drawn destination-point permutation for every training-seed, "
                "probe-subset, and checkpoint-pair baseline record."
            ),
            "empirical_records": total_records,
            "ordinary_records": ordinary_records,
            "permutation_null_records": permutation_records,
        },
        "integrity": integrity,
        "source_files": source_files,
        "phase_conventions": {
            convention: _summarise_convention(
                runs,
                convention,
                fixed_window,
                alignment_threshold,
                alignment_offsets,
            )
            for convention in ("fixed", "event_aligned")
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    headers = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"


def _macro(name: str, value: Any) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def write_cross_seed_hodge_outputs(
    summary: dict[str, Any],
    output_dir: str | Path,
    paper_generated_dir: str | Path | None = None,
) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    summary_path = write_json(output_dir / "cross_seed_hodge_summary.json", summary)
    contrast_rows = []
    seed_phase_rows = []
    for convention, result in summary["phase_conventions"].items():
        for row in result["seed_setting_contrasts"]:
            contrast_rows.append({"phase_convention": convention, **row})
        for row in result["baseline_by_seed_phase"]:
            for phase in ("pre", "transition", "post"):
                seed_phase_rows.append(
                    {
                        "phase_convention": convention,
                        "training_seed": row["training_seed"],
                        "phase": phase,
                        **row[phase],
                    }
                )
    contrasts_path = _write_csv(output_dir / "seed_setting_contrasts.csv", contrast_rows)
    phases_path = _write_csv(output_dir / "baseline_seed_phase_means.csv", seed_phase_rows)

    generated_dir = ensure_dir(paper_generated_dir or output_dir)
    fixed = summary["phase_conventions"]["fixed"]
    event = summary["phase_conventions"]["event_aligned"]
    fixed_change = fixed["seed_level_inference"][
        "baseline_post_minus_transition_coexact_minus_exact"
    ]
    event_change = event["seed_level_inference"][
        "baseline_post_minus_transition_coexact_minus_exact"
    ]
    fixed_null_exact = fixed["seed_level_inference"]["post_actual_minus_null_exact"]
    fixed_null_coexact = fixed["seed_level_inference"]["post_actual_minus_null_coexact"]
    fixed_counts = fixed["direction_counts"]
    event_counts = event["direction_counts"]

    macros = [
        "% Generated by reproducibility/Grokking/Analysis/12_hodge_cross_seed_summary.py.",
        _macro("HodgeTrainingSeeds", summary["design"]["n_training_seeds"]),
        _macro(
            "HodgeSensitivityComparisons",
            fixed_counts["seed_setting_shift_toward_exact"]["total"],
        ),
        _macro(
            "HodgeFixedNegativeComparisons",
            fixed_counts["seed_setting_shift_toward_exact"]["count"],
        ),
        _macro(
            "HodgeFixedPostExactComparisons",
            fixed_counts["seed_setting_post_exact_exceeds_coexact"]["count"],
        ),
        _macro(
            "HodgeEventNegativeComparisons",
            event_counts["seed_setting_shift_toward_exact"]["count"],
        ),
        _macro(
            "HodgeEventPostExactComparisons",
            event_counts["seed_setting_post_exact_exceeds_coexact"]["count"],
        ),
        _macro("HodgeFixedChange", _fmt(fixed_change["mean"])),
        _macro("HodgeFixedChangeLow", _fmt(fixed_change["interval"][0])),
        _macro("HodgeFixedChangeHigh", _fmt(fixed_change["interval"][1])),
        _macro("HodgeEventChange", _fmt(event_change["mean"])),
        _macro("HodgeEventChangeLow", _fmt(event_change["interval"][0])),
        _macro("HodgeEventChangeHigh", _fmt(event_change["interval"][1])),
        _macro("HodgePostNullExact", _fmt(fixed_null_exact["mean"])),
        _macro("HodgePostNullExactLow", _fmt(fixed_null_exact["interval"][0])),
        _macro("HodgePostNullExactHigh", _fmt(fixed_null_exact["interval"][1])),
        _macro("HodgePostNullCoexact", _fmt(fixed_null_coexact["mean"])),
        _macro("HodgePostNullCoexactLow", _fmt(fixed_null_coexact["interval"][0])),
        _macro("HodgePostNullCoexactHigh", _fmt(fixed_null_coexact["interval"][1])),
    ]
    for phase, prefix in (("pre", "Pre"), ("transition", "Transition"), ("post", "Post")):
        components = (("exact", "Exact"), ("coexact", "Coexact"), ("harmonic", "Harmonic"))
        for component, component_name in components:
            macros.append(
                _macro(
                    f"HodgeFixed{prefix}{component_name}",
                    _fmt(fixed["baseline_phase_summary"][phase][component]["mean"]),
                )
            )
    macros_path = generated_dir / "eight_seed_hodge_macros.tex"
    macros_path.write_text("\n".join(macros) + "\n", encoding="utf-8")

    baseline_table = [
        "% Generated by reproducibility/Grokking/Analysis/12_hodge_cross_seed_summary.py.",
        r"\newcommand{\HodgeBaselineRows}{%",
        r"\addedtext{Pre-grokking} & \addedtext{\HodgeFixedPreExact} & "
        r"\addedtext{\HodgeFixedPreCoexact} & \addedtext{\HodgeFixedPreHarmonic} \\",
        r"\addedtext{Transition} & \addedtext{\HodgeFixedTransitionExact} & "
        r"\addedtext{\HodgeFixedTransitionCoexact} & "
        r"\addedtext{\HodgeFixedTransitionHarmonic} \\",
        r"\addedtext{Post-grokking} & \addedtext{\HodgeFixedPostExact} & "
        r"\addedtext{\HodgeFixedPostCoexact} & \addedtext{\HodgeFixedPostHarmonic} \\",
        "}",
    ]
    baseline_path = generated_dir / "eight_seed_hodge_baseline_rows.tex"
    baseline_path.write_text("\n".join(baseline_table) + "\n", encoding="utf-8")

    sensitivity_table = [
        "% Generated by reproducibility/Grokking/Analysis/12_hodge_cross_seed_summary.py.",
        r"\newcommand{\HodgePhaseSensitivityRows}{%",
        r"\addedtext{Fixed epochs} & \addedtext{$\HodgeFixedChange$} & "
        r"\addedtext{$[\HodgeFixedChangeLow,\HodgeFixedChangeHigh]$} & "
        r"\addedtext{\HodgeFixedNegativeComparisons/\HodgeSensitivityComparisons} & "
        r"\addedtext{\HodgeFixedPostExactComparisons/\HodgeSensitivityComparisons} \\",
        r"\addedtext{Seed-relative} & \addedtext{$\HodgeEventChange$} & "
        r"\addedtext{$[\HodgeEventChangeLow,\HodgeEventChangeHigh]$} & "
        r"\addedtext{\HodgeEventNegativeComparisons/\HodgeSensitivityComparisons} & "
        r"\addedtext{\HodgeEventPostExactComparisons/\HodgeSensitivityComparisons} \\",
        "}",
    ]
    sensitivity_path = generated_dir / "eight_seed_hodge_phase_sensitivity_rows.tex"
    sensitivity_path.write_text("\n".join(sensitivity_table) + "\n", encoding="utf-8")

    output_manifest = []
    for path in (
        summary_path,
        contrasts_path,
        phases_path,
        macros_path,
        baseline_path,
        sensitivity_path,
    ):
        manifest_name = path.name if path.parent == output_dir else f"generated/{path.name}"
        output_manifest.append(
            {"path": manifest_name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    manifest_path = write_json(output_dir / "generated_manifest.json", output_manifest)
    return {
        "summary": str(summary_path),
        "contrasts_csv": str(contrasts_path),
        "phase_csv": str(phases_path),
        "macros": str(macros_path),
        "baseline_rows": str(baseline_path),
        "phase_sensitivity_rows": str(sensitivity_path),
        "manifest": str(manifest_path),
    }
