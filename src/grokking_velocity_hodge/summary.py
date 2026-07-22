import numpy as np

from .config import ExperimentConfig


def effective_dimension(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    cov = np.cov(x.T)
    evals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    s1 = float(evals.sum())
    s2 = float(np.sum(evals**2))
    return float((s1 * s1 / s2) if s2 > 0 else 0.0)


def summarise_transition_series(
    midpoints,
    values,
    transition_start: float | None = None,
    transition_end: float | None = None,
) -> dict[str, float | None]:
    config = ExperimentConfig.from_environment()
    transition_start = config.transition_start if transition_start is None else transition_start
    transition_end = config.transition_end if transition_end is None else transition_end
    mids = np.asarray(midpoints, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    transition = vals[(mids >= transition_start) & (mids <= transition_end)]
    post = vals[mids > transition_end]
    initial_midpoint = config.save_every / 2
    initial = vals[mids == initial_midpoint]
    post_mean = float(np.mean(post)) if len(post) else None
    transition_peak = float(np.max(transition)) if len(transition) else None
    transition_mean = float(np.mean(transition)) if len(transition) else None
    return {
        "initial_0_500": float(initial[0]) if len(initial) else None,
        "transition_peak": transition_peak,
        "transition_mean": transition_mean,
        "post_mean": post_mean,
        "transition_peak_over_post_mean": (
            float(transition_peak / post_mean)
            if transition_peak is not None and post_mean is not None and post_mean > 0
            else None
        ),
        "transition_mean_over_post_mean": (
            float(transition_mean / post_mean)
            if transition_mean is not None and post_mean is not None and post_mean > 0
            else None
        ),
    }


def mean_sd(values, ignore_none: bool = False, include_n: bool = False) -> dict[str, float | int | None]:
    raw = list(values)
    clean = [float(v) for v in raw if v is not None] if ignore_none else [float(v) for v in raw]
    if clean:
        arr = np.asarray(clean, dtype=np.float64)
        out: dict[str, float | int | None] = {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
    else:
        out = {"mean": None if ignore_none else float("nan"), "sd": None if ignore_none else 0.0}
    if include_n:
        out["n"] = int(len(clean))
    return out
