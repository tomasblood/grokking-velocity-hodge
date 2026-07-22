"""Helpers for auditing cached and freshly reproduced BW series."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from .summary import summarise_transition_series


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_bw_series(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return payload["bw_distances_consecutive"]


def compare_bw_files(cached_path: str | Path, rerun_path: str | Path) -> dict:
    cached = load_bw_series(cached_path)
    rerun = load_bw_series(rerun_path)
    cached_midpoints = np.asarray(cached["midpoint_epochs"], dtype=float)
    rerun_midpoints = np.asarray(rerun["midpoint_epochs"], dtype=float)
    cached_distances = np.asarray(cached["distances"], dtype=float)
    rerun_distances = np.asarray(rerun["distances"], dtype=float)

    same_grid = np.array_equal(cached_midpoints, rerun_midpoints)
    if cached_distances.shape != rerun_distances.shape:
        max_difference = None
    else:
        max_difference = float(np.max(np.abs(cached_distances - rerun_distances)))

    cached_summary = summarise_transition_series(cached_midpoints, cached_distances)
    rerun_summary = summarise_transition_series(rerun_midpoints, rerun_distances)
    cached_ratio = cached_summary["transition_peak_over_post_mean"]
    rerun_ratio = rerun_summary["transition_peak_over_post_mean"]
    ratio_difference = (
        abs(float(cached_ratio) - float(rerun_ratio))
        if cached_ratio is not None and rerun_ratio is not None
        else None
    )
    conclusion_stable = same_grid and ratio_difference is not None and ratio_difference <= 0.02

    return {
        "cached": {"path": str(cached_path), "sha256": file_sha256(cached_path), "summary": cached_summary},
        "rerun": {"path": str(rerun_path), "sha256": file_sha256(rerun_path), "summary": rerun_summary},
        "same_midpoint_grid": same_grid,
        "series_exactly_equal": same_grid and np.array_equal(cached_distances, rerun_distances),
        "max_absolute_distance_difference": max_difference,
        "transition_peak_ratio_difference": ratio_difference,
        "conclusion_stable": conclusion_stable,
    }
