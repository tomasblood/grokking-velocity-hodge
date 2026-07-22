"""Probe, parameter, and permutation robustness for velocity Hodge fractions."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import diffusion_geometry as dg
import numpy as np
from sklearn.decomposition import PCA

from .config import ExperimentConfig
from .runtime import hodge_decompose_velocity


def _tuple_env(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    return default if not value else tuple(int(item.strip()) for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class HodgeSweepConfig:
    pca_dims: tuple[int, ...] = (5, 10, 20)
    knn_values: tuple[int, ...] = (10, 15, 20)
    basis_values: tuple[int, ...] = (30, 50, 75)
    subset_size: int = 400
    subset_seeds: tuple[int, ...] = (1101, 2202, 3303, 4404)
    pairs_per_phase: int = 3

    @classmethod
    def from_environment(cls) -> "HodgeSweepConfig":
        return cls(
            pca_dims=_tuple_env("GROKKING_HODGE_SWEEP_PCA_DIMS", cls.pca_dims),
            knn_values=_tuple_env("GROKKING_HODGE_SWEEP_KNN", cls.knn_values),
            basis_values=_tuple_env("GROKKING_HODGE_SWEEP_BASES", cls.basis_values),
            subset_size=int(os.environ.get("GROKKING_HODGE_SWEEP_SUBSET_SIZE", cls.subset_size)),
            subset_seeds=_tuple_env("GROKKING_HODGE_SWEEP_SUBSET_SEEDS", cls.subset_seeds),
            pairs_per_phase=int(os.environ.get("GROKKING_HODGE_SWEEP_PAIRS_PER_PHASE", cls.pairs_per_phase)),
        )


def one_at_a_time_settings(experiment: ExperimentConfig, sweep: HodgeSweepConfig) -> list[dict[str, int]]:
    baseline = {
        "pca_dim": experiment.hodge_pca_dim,
        "knn": experiment.knn,
        "n_basis": experiment.hodge_basis,
    }
    settings = {tuple(baseline.items()): baseline}
    for key, values in (
        ("pca_dim", sweep.pca_dims),
        ("knn", sweep.knn_values),
        ("n_basis", sweep.basis_values),
    ):
        for value in values:
            setting = baseline | {key: value}
            settings[tuple(setting.items())] = setting
    return list(settings.values())


def representative_pairs(
    epochs: list[int],
    transition_start: int,
    transition_end: int,
    pairs_per_phase: int,
) -> list[dict]:
    grouped = {"pre": [], "transition": [], "post": []}
    for epoch_a, epoch_b in zip(epochs[:-1], epochs[1:]):
        midpoint = (epoch_a + epoch_b) / 2.0
        phase = (
            "pre" if midpoint < transition_start else "post" if midpoint > transition_end else "transition"
        )
        grouped[phase].append((epoch_a, epoch_b, midpoint))

    selected = []
    for phase, pairs in grouped.items():
        if not pairs:
            continue
        indices = np.linspace(0, len(pairs) - 1, min(pairs_per_phase, len(pairs)), dtype=int)
        for index in np.unique(indices):
            epoch_a, epoch_b, midpoint = pairs[int(index)]
            selected.append({"phase": phase, "epoch_a": epoch_a, "epoch_b": epoch_b, "midpoint": midpoint})
    return selected


def _aggregate(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        key = (
            record["phase"],
            record["pca_dim"],
            record["knn"],
            record["n_basis"],
            record["permuted_correspondence"],
        )
        grouped.setdefault(key, []).append(record)

    output = []
    for key, rows in grouped.items():
        item = dict(zip(("phase", "pca_dim", "knn", "n_basis", "permuted_correspondence"), key))
        item["n"] = len(rows)
        for component in ("exact", "coexact", "harmonic"):
            values = np.asarray([row[component] for row in rows], dtype=float)
            item[f"mean_{component}"] = float(values.mean())
            item[f"sd_{component}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        item["mean_coexact_minus_exact"] = item["mean_coexact"] - item["mean_exact"]
        output.append(item)
    return output


def run_hodge_robustness(
    activation_dir: str | Path,
    epochs: list[int],
    experiment: ExperimentConfig,
    sweep: HodgeSweepConfig,
) -> dict:
    activation_dir = Path(activation_dir)
    pairs = representative_pairs(
        epochs,
        experiment.transition_start,
        experiment.transition_end,
        sweep.pairs_per_phase,
    )
    settings = one_at_a_time_settings(experiment, sweep)
    baseline = {
        "pca_dim": experiment.hodge_pca_dim,
        "knn": experiment.knn,
        "n_basis": experiment.hodge_basis,
    }

    n_points = np.load(activation_dir / f"act_{epochs[0]}.npy", mmap_mode="r").shape[0]
    subset_size = min(sweep.subset_size, n_points)
    required_epochs = sorted({value for pair in pairs for value in (pair["epoch_a"], pair["epoch_b"])})
    records = []

    for subset_seed in sweep.subset_seeds:
        rng = np.random.default_rng(subset_seed)
        subset = np.sort(rng.choice(n_points, subset_size, replace=False))
        activations = {
            epoch: np.load(activation_dir / f"act_{epoch}.npy").astype(np.float64)[subset]
            for epoch in required_epochs
        }
        for setting in settings:
            for pair in pairs:
                source = activations[pair["epoch_a"]]
                destination = activations[pair["epoch_b"]]
                decomposition = hodge_decompose_velocity(
                    source,
                    destination - source,
                    PCA,
                    dg,
                    pca_solver=experiment.pca_solver,
                    **setting,
                )
                records.append(
                    pair
                    | setting
                    | decomposition
                    | {"subset_seed": subset_seed, "permuted_correspondence": False}
                )

                if setting == baseline:
                    permutation = rng.permutation(len(destination))
                    null = hodge_decompose_velocity(
                        source,
                        destination[permutation] - source,
                        PCA,
                        dg,
                        pca_solver=experiment.pca_solver,
                        **setting,
                    )
                    records.append(
                        pair | setting | null | {"subset_seed": subset_seed, "permuted_correspondence": True}
                    )

    return {
        "experiment_config": experiment.to_dict(),
        "sweep_config": asdict(sweep),
        "settings": settings,
        "pairs": pairs,
        "records": records,
        "aggregate": _aggregate(records),
    }
