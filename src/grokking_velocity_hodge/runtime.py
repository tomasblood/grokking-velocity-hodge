import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, hsv_to_rgb, rgb_to_hsv
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig

# --- Plot constants

# Shared plotting palette used by the chapter figures.
TEXT_COLOR = "#1D1D1D"
GRID_COLOR = "#E6E8EB"
VALIDATION_COLOR = "#2A9D8F"
MAIN_COLOR = "#2A9D8F"
ACCENT_COLOR = "#4C78A8"
SECONDARY_COLOR = "#D1884F"
GREY_COLOR = "#5F6670"

GROKKING_SHADE = "#F2C879"
GROKKING_ALPHA = 0.16


# --- Notebook parameters and paths


# find databricks helpers when present
def get_dbutils_or_none():
    try:
        return dbutils  # type: ignore[name-defined]
    except NameError:
        pass
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and "dbutils" in shell.user_ns:
            return shell.user_ns["dbutils"]
    except Exception:
        shell = None
    return None


# this lets the same notebook use databricks widgets or local env vars
def notebook_param(name: str, default: str = "") -> str:
    dbutils_obj = get_dbutils_or_none()
    if dbutils_obj is not None:
        try:
            value = dbutils_obj.widgets.get(name)
            if value not in (None, ""):
                return value
        except Exception:
            value = ""
    return os.environ.get(name, default)


def _find_repo_root(start):
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


# this decides where data, results and figures are written
# it lets the same code run against dbfs or a local root
def thesis_root() -> Path:
    override = notebook_param("THESIS_DATA_ROOT", "").strip()
    if override:
        return Path(override)
    if Path("/dbfs").exists():
        return Path(notebook_param("THESIS_DBFS_ROOT", "/dbfs/FileStore/thesis"))
    if "__file__" in globals():
        return _find_repo_root(Path(__file__).resolve())
    return Path.cwd()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def grokking_figure_subdir(chart_variant: str) -> str:
    chart_variant = (chart_variant or "").strip()
    if chart_variant in {"", "original", "legacy"}:
        return "grokking"
    return f"grokking_{chart_variant}"


@dataclass(frozen=True)
class GrokkingRuntime:
    root: Path
    chart_variant: str
    figure_subdir: str
    figure_dir: Path
    activation_dir: Path

    def result_dir(self, name: str) -> Path:
        return ensure_dir(self.root / "results" / name)


def configure_grokking_runtime(chart_variant: str | None = None) -> GrokkingRuntime:
    root = thesis_root()
    chart_variant = (
        chart_variant if chart_variant is not None else notebook_param("GROKKING_CHART_VARIANT", "new_charts")
    ).strip()
    fig_subdir = grokking_figure_subdir(chart_variant)
    figure_dir = Path(notebook_param("GROKKING_FIGURE_DIR", str(root / "figures" / fig_subdir)))
    activation_dir = Path(
        notebook_param("GROKKING_ACTIVATION_DIR", str(root / "results" / "grokking_acts_v6"))
    )
    return GrokkingRuntime(
        root=root,
        chart_variant=chart_variant,
        figure_subdir=fig_subdir,
        figure_dir=ensure_dir(figure_dir),
        activation_dir=activation_dir,
    )


# --- Figure style helpers


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.15,
            "axes.edgecolor": "#2F3437",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.45,
            "grid.linestyle": ":",
            "grid.alpha": 0.95,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.45,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.prop_cycle": plt.cycler(
                color=[MAIN_COLOR, ACCENT_COLOR, VALIDATION_COLOR, SECONDARY_COLOR, GREY_COLOR]
            ),
        }
    )


def shade_grokking_window(ax, label: bool = False) -> None:
    config = ExperimentConfig.from_environment()
    ax.axvspan(
        config.transition_start,
        config.transition_end,
        color=GROKKING_SHADE,
        alpha=GROKKING_ALPHA,
        zorder=0,
        label="grokking window" if label else None,
    )


def dimmed_phase_cmap(name: str = "hsv", saturation: float = 0.62, value: float = 0.78):
    colors = plt.get_cmap(name)(np.linspace(0.0, 1.0, 256))
    hsv = rgb_to_hsv(colors[:, :3])
    hsv[:, 1] = np.clip(hsv[:, 1] * saturation, 0.0, 1.0)
    hsv[:, 2] = np.clip(hsv[:, 2] * value, 0.0, 1.0)
    colors[:, :3] = hsv_to_rgb(hsv)
    return ListedColormap(colors, name=f"{name}_dimmed")


# --- Data and JSON helpers


def load_training_meta(act_dir: str | Path) -> dict:
    with (Path(act_dir) / "training.json").open("r", encoding="utf-8") as f:
        training = json.load(f)
    if "val_accs" not in training and "test_accs" in training:
        training["val_accs"] = training["test_accs"]
    return training


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def write_json(path: str | Path, payload: object, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=indent, allow_nan=False)
    return path


def edge_padded_moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window or window <= 1:
        return arr.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(arr, (left, right), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


# --- Bures Wasserstein matrix geometry
def real_sqrtm(a: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(a)
    evals = np.maximum(evals, 0.0)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T


# bw is always applied after operators are turned into covariance matrices
# outputs are symmetrised to avoid small numerical asymmetries
def bw_distance(s0: np.ndarray, s1: np.ndarray) -> float:
    sqrt_s0 = real_sqrtm(s0)
    inner = sqrt_s0 @ s1 @ sqrt_s0
    sqrt_inner = real_sqrtm(inner)
    d_sq = np.trace(s0) + np.trace(s1) - 2.0 * np.trace(sqrt_inner)
    return float(np.sqrt(max(d_sq, 0.0)))


# --- Graph and spectral operator helpers


def build_normalised_laplacian(x: np.ndarray, k: int = 15) -> np.ndarray:
    n = x.shape[0]
    if n == 0:
        raise ValueError("build_normalised_laplacian requires at least one point")
    n_neighbors = min(max(int(k), 0) + 1, n)
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(x)
    dists, indices = nn.kneighbors(x)
    weights = np.zeros((n, n))
    for i in range(n):
        sigma_i = dists[i, -1] + 1e-10
        for j_idx in range(n_neighbors):
            j = indices[i, j_idx]
            if i != j:
                d_ij = dists[i, j_idx]
                w = np.exp(-(d_ij**2) / (2.0 * sigma_i**2))
                weights[i, j] = max(weights[i, j], w)
                weights[j, i] = max(weights[j, i], w)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(weights.sum(axis=1), 1e-10)))
    laplacian = np.eye(n) - d_inv_sqrt @ weights @ d_inv_sqrt
    return 0.5 * (laplacian + laplacian.T)


def laplacian_spectrum(laplacian: np.ndarray, k: int = 30) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = eigh(laplacian)
    evals = np.maximum(evals, 0.0)
    return evals[1 : k + 1], evecs[:, 1 : k + 1]


# --- Spectral covariance coordinates


# project the current covariance into the reference eigenbasis
def covariance_in_reference_basis(
    evals: np.ndarray,
    evecs: np.ndarray,
    ref_basis: np.ndarray,
    eps: float = 0.01,
    floor: float = 1e-10,
) -> np.ndarray:
    cov_evals = 1.0 / (np.asarray(evals, dtype=float) + eps)
    change_of_basis = ref_basis.T @ evecs
    sigma = change_of_basis @ np.diag(cov_evals) @ change_of_basis.T
    sigma = 0.5 * (sigma + sigma.T)
    if floor > 0:
        sigma = sigma + floor * np.eye(sigma.shape[0])
    return sigma


def heat_kernel_in_reference_basis(
    evals: np.ndarray,
    evecs: np.ndarray,
    ref_basis: np.ndarray,
    tau: float,
    floor: float = 1e-10,
) -> np.ndarray:
    heat_evals = np.exp(-tau * np.asarray(evals, dtype=float))
    change_of_basis = ref_basis.T @ evecs
    heat = change_of_basis @ np.diag(heat_evals) @ change_of_basis.T
    heat = 0.5 * (heat + heat.T)
    if floor > 0:
        heat = heat + floor * np.eye(heat.shape[0])
    return heat


# --- Fourier and circular coordinate helpers


def dominant_fourier_frequency(x: np.ndarray, labels: np.ndarray, p: int = 113) -> tuple[int, np.ndarray]:
    powers = np.zeros(p)
    for k in range(p):
        phases = np.exp(-2j * np.pi * k * labels / p)
        c_k = x.T @ phases
        powers[k] = np.sum(np.abs(c_k) ** 2)
    powers[0] = 0
    k_raw = int(np.argmax(powers))
    return min(k_raw, p - k_raw), powers


def circular_correlation(theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    c_fwd = abs(np.mean(np.exp(1j * (theta_a - theta_b))))
    c_rev = abs(np.mean(np.exp(1j * (theta_a + theta_b))))
    return float(max(c_fwd, c_rev))


def fourier_ridge_projection(
    x: np.ndarray,
    labels: np.ndarray,
    k: int,
    p: int = 113,
    alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    cos_t = np.cos(2 * np.pi * k * labels / p)
    sin_t = np.sin(2 * np.pi * k * labels / p)
    r_cos = Ridge(alpha=alpha).fit(x_scaled, cos_t)
    r_sin = Ridge(alpha=alpha).fit(x_scaled, sin_t)
    z_cos = r_cos.predict(x_scaled)
    z_sin = r_sin.predict(x_scaled)
    r2 = (r_cos.score(x_scaled, cos_t) + r_sin.score(x_scaled, sin_t)) / 2
    return z_cos, z_sin, float(r2)


def best_fourier_ridge_frequency(
    x: np.ndarray,
    labels: np.ndarray,
    p: int = 113,
    alpha: float = 10.0,
) -> tuple[int, float, list[tuple[int, float]]]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    best_k, best_r2 = 0, -1.0
    scored = []
    for k in range(1, p // 2 + 1):
        cos_t = np.cos(2 * np.pi * k * labels / p)
        sin_t = np.sin(2 * np.pi * k * labels / p)
        r_cos = Ridge(alpha=alpha).fit(x_scaled, cos_t)
        r_sin = Ridge(alpha=alpha).fit(x_scaled, sin_t)
        r2 = (r_cos.score(x_scaled, cos_t) + r_sin.score(x_scaled, sin_t)) / 2
        scored.append((k, float(r2)))
        if r2 > best_r2:
            best_k, best_r2 = k, float(r2)
    scored.sort(key=lambda item: item[1], reverse=True)
    return best_k, best_r2, scored


def fourier_circular_coordinate(
    x: np.ndarray,
    labels: np.ndarray,
    k: int,
    p: int = 113,
) -> np.ndarray:
    cos_k = np.cos(2 * np.pi * k * labels / p)
    sin_k = np.sin(2 * np.pi * k * labels / p)
    a_cos = x.T @ cos_k
    a_sin = x.T @ sin_k
    a_cos = a_cos / max(np.linalg.norm(a_cos), 1e-12)
    a_sin = a_sin / max(np.linalg.norm(a_sin), 1e-12)
    return np.arctan2(x @ a_sin, x @ a_cos) % (2 * np.pi)


def dg_circular_coordinate_from_rep(
    x_rep: np.ndarray,
    labels: np.ndarray,
    k_dom: int,
    dg_module,
    p: int = 113,
    knn: int = 15,
    n_basis: int = 30,
    n_eigpairs: int = 15,
) -> tuple[np.ndarray | None, float, tuple[int, int], int]:
    model = dg_module.DiffusionGeometry.from_point_cloud(
        x_rep.astype(np.float64),
        knn_kernel=knn,
        n_function_basis=n_basis,
    )
    phi = model.function_basis

    test_freqs = set()
    for dk in range(-3, 4):
        kk = (k_dom + dk) % p
        test_freqs.add(min(kk, p - kk))
    for k in range(1, p // 2 + 1):
        test_freqs.add(k)

    best_corr = 0.0
    best_theta = None
    best_pair = (0, 0)
    best_k = 0

    n_funcs = min(n_eigpairs, phi.shape[1])
    for i in range(1, n_funcs):
        for j in range(i + 1, n_funcs):
            theta_ij = np.arctan2(phi[:, j], phi[:, i]) % (2 * np.pi)
            for k in test_freqs:
                theta_k = (labels * k * 2 * np.pi / p) % (2 * np.pi)
                corr = circular_correlation(theta_ij, theta_k)
                if corr > best_corr:
                    best_corr = corr
                    best_theta = theta_ij
                    best_pair = (i, j)
                    best_k = k

    return best_theta, best_corr, best_pair, best_k


def dg_circular_coordinate(
    x: np.ndarray,
    labels: np.ndarray,
    k_dom: int,
    pca_cls,
    dg_module,
    p: int = 113,
    pca_dim: int = 10,
    pca_solver: str = "full",
    knn: int = 15,
    n_basis: int = 30,
    n_eigpairs: int = 15,
):
    pca = pca_cls(n_components=pca_dim, svd_solver=pca_solver)
    x_red = pca.fit_transform(x)
    return dg_circular_coordinate_from_rep(
        x_red,
        labels,
        k_dom,
        dg_module,
        p=p,
        knn=knn,
        n_basis=n_basis,
        n_eigpairs=n_eigpairs,
    )


# --- Sampling and Hodge decomposition helpers


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    n = min(int(n), len(points))
    if n <= 0:
        return np.array([], dtype=int)
    indices = [0]
    distances = np.full(len(points), np.inf)
    for _ in range(n - 1):
        distances = np.minimum(distances, np.linalg.norm(points - points[indices[-1]], axis=1))
        indices.append(int(np.argmax(distances)))
    return np.array(indices, dtype=int)


# turns activation velocities into a dg one form before hodge decomp using the diffusion geometry package
def hodge_decompose_velocity(
    x: np.ndarray,
    velocity: np.ndarray,
    pca_cls,
    dg_module,
    pca_dim: int = 10,
    pca_solver: str = "full",
    knn: int = 15,
    n_basis: int = 50,
) -> dict[str, float]:
    pca = pca_cls(n_components=pca_dim, svd_solver=pca_solver)
    x_red = pca.fit_transform(x)
    velocity_red = velocity @ pca.components_.T
    model = dg_module.DiffusionGeometry.from_point_cloud(
        x_red.astype(np.float64),
        knn_kernel=knn,
        n_function_basis=n_basis,
    )
    omega = model.form(velocity_red.astype(np.float64), degree=1)
    exact_f, coexact_g, harmonic_h = omega.hodge_decomposition()
    exact_1 = exact_f.d()
    coexact_1 = coexact_g.codifferential()
    e_exact = float(exact_1.norm() ** 2)
    e_coexact = float(coexact_1.norm() ** 2)
    e_harmonic = float(harmonic_h.norm() ** 2)
    e_total = float(omega.norm() ** 2)
    if e_total < 1e-16:
        return {"exact": 0.0, "coexact": 0.0, "harmonic": 0.0, "total_energy": 0.0}
    e_sum = e_exact + e_coexact + e_harmonic
    return {
        "exact": float(e_exact / e_sum),
        "coexact": float(e_coexact / e_sum),
        "harmonic": float(e_harmonic / e_sum),
        "total_energy": float(e_total),
    }


# End of shared runtime helpers.
