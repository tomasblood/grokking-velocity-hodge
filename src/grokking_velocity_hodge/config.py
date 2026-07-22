import os
from dataclasses import asdict, dataclass
from typing import Any


def _env(name: str, default: Any, cast):
    return cast(os.environ.get(name, default))


def _env_tuple(name: str, default: tuple, cast) -> tuple:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class ExperimentConfig:
    """Single source of truth for training and analysis parameters."""

    modulus: int = 113
    d_model: int = 128
    max_epoch: int = 25_000
    save_every: int = 500
    train_fraction: float = 0.3
    probe_size: int = 500
    data_seed: int = 598
    probe_seed: int = 42

    pca_dim: int = 20
    pca_solver: str = "full"
    knn: int = 15
    spectral_components: int = 30
    resolvent_epsilon: float = 0.01

    hodge_pca_dim: int = 10
    hodge_basis: int = 50
    hodge_quiver_points: int = 60
    circular_pca_dim: int = 10
    circular_basis: int = 30
    circular_eigenpairs: int = 15
    heat_scales: tuple[float, ...] = (0.1, 1.0, 10.0)
    probe_subset_size: int = 400
    probe_subset_seeds: tuple[int, ...] = (1101, 2202, 3303, 4404)

    transition_start: int = 1_500
    transition_end: int = 4_000
    event_transition_delta: tuple[float, ...] = (-1_500.0, 1_000.0)
    event_x_limits: tuple[float, ...] = (-3_000.0, 10_000.0)
    event_grid_step: float = 500.0
    event_alignment_threshold: float = 0.5

    @classmethod
    def from_environment(cls) -> "ExperimentConfig":
        return cls(
            modulus=_env("GROKKING_P", cls.modulus, int),
            d_model=_env("GROKKING_D_MODEL", cls.d_model, int),
            max_epoch=_env("GROKKING_N_EPOCHS", cls.max_epoch, int),
            save_every=_env("GROKKING_SAVE_EVERY", cls.save_every, int),
            train_fraction=_env("GROKKING_TRAIN_FRAC", cls.train_fraction, float),
            probe_size=_env("GROKKING_N_SUB", cls.probe_size, int),
            data_seed=_env("GROKKING_DATA_SEED", cls.data_seed, int),
            probe_seed=_env("GROKKING_PROBE_SEED", cls.probe_seed, int),
            pca_dim=_env("GROKKING_PCA_DIM", cls.pca_dim, int),
            pca_solver=os.environ.get("GROKKING_PCA_SOLVER", cls.pca_solver),
            knn=_env("GROKKING_KNN", cls.knn, int),
            spectral_components=_env("GROKKING_K_SPEC", cls.spectral_components, int),
            resolvent_epsilon=_env("GROKKING_RESOLVENT_EPS", cls.resolvent_epsilon, float),
            hodge_pca_dim=_env("GROKKING_HODGE_PCA_DIM", cls.hodge_pca_dim, int),
            hodge_basis=_env("GROKKING_HODGE_BASIS", cls.hodge_basis, int),
            hodge_quiver_points=_env("GROKKING_HODGE_QUIVER_POINTS", cls.hodge_quiver_points, int),
            circular_pca_dim=_env("GROKKING_CIRCULAR_PCA_DIM", cls.circular_pca_dim, int),
            circular_basis=_env("GROKKING_CIRCULAR_BASIS", cls.circular_basis, int),
            circular_eigenpairs=_env(
                "GROKKING_CIRCULAR_EIGENPAIRS",
                cls.circular_eigenpairs,
                int,
            ),
            heat_scales=_env_tuple("GROKKING_HEAT_SCALES", cls.heat_scales, float),
            probe_subset_size=_env("GROKKING_PROBE_SUBSET_SIZE", cls.probe_subset_size, int),
            probe_subset_seeds=_env_tuple(
                "GROKKING_PROBE_SUBSET_SEEDS",
                cls.probe_subset_seeds,
                int,
            ),
            transition_start=_env("GROKKING_TRANSITION_START", cls.transition_start, int),
            transition_end=_env("GROKKING_TRANSITION_END", cls.transition_end, int),
            event_transition_delta=_env_tuple(
                "GROKKING_EVENT_TRANSITION_DELTA", cls.event_transition_delta, float
            ),
            event_x_limits=_env_tuple("GROKKING_EVENT_X_LIMITS", cls.event_x_limits, float),
            event_grid_step=_env("GROKKING_EVENT_GRID_STEP", cls.event_grid_step, float),
            event_alignment_threshold=_env(
                "GROKKING_EVENT_ALIGNMENT_THRESHOLD", cls.event_alignment_threshold, float
            ),
        )

    @property
    def epochs(self) -> list[int]:
        return list(range(0, self.max_epoch + 1, self.save_every))

    def checkpoint_epochs(self, training: dict[str, Any] | None = None) -> list[int]:
        if training:
            saved = training.get("saved_epochs") or training.get("epochs")
            if saved:
                return [int(epoch) for epoch in saved]
        return self.epochs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
