from dataclasses import dataclass
from pathlib import Path
import json
import os
import shutil
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# --- Plot constants

# Shared plotting palette used by the chapter figures.
TEXT_COLOR = "#1D1D1D"
GRID_COLOR = "#E6E8EB"
VALIDATION_COLOR = "#2A9D8F"
MAIN_COLOR = "#2A9D8F"
ACCENT_COLOR = "#4C78A8"
SECONDARY_COLOR = "#D1884F"
GREY_COLOR = "#5F6670"
SKY_COLOR = "#7AA6C2"
EIGENVALUE_COLORS = [ACCENT_COLOR, MAIN_COLOR, VALIDATION_COLOR, SECONDARY_COLOR, SKY_COLOR]

GROKKING_START = 1500
GROKKING_END = 4000
GROKKING_SHADE = "#F2C879"
GROKKING_ALPHA = 0.16


# --- Notebook parameters and paths

# find databricks helpers when present
def get_dbutils_or_none():
    try:
        return dbutils  # type: ignore[name-defined]
    except NameError:
        dbutils_obj = None
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


def parse_years(spec: str) -> list[int]:
    spec = str(spec).strip()
    if not spec:
        return []
    if "-" in spec:
        start, end = [int(x.strip()) for x in spec.split("-", 1)]
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


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


# --- File output helpers

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def uses_dbfs_fuse(path: str | Path) -> bool:
    return str(path).startswith("/dbfs/")


# we had databricks/dbfs writes that could leave partial outputs if a notebook died halfway through
# this writes to /tmp first then copies to the target
def _tmp_write_path(path, suffix):
    tmp_dir = ensure_dir(Path(notebook_param("THESIS_TMP_WRITE_DIR", "/tmp/thesis_safe_writes")))
    return tmp_dir / f"{path.stem}_{uuid.uuid4().hex}{suffix}"


def _copy_tmp_to_target(tmp_path: Path, target_path: Path) -> Path:
    ensure_dir(target_path.parent)
    shutil.copy2(tmp_path, target_path)
    try:
        tmp_path.unlink()
    except OSError:
        pass
    return target_path


# dbfs writes use temp files
def save_npz_compressed(path: str | Path, **arrays) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if uses_dbfs_fuse(path):
        tmp_path = _tmp_write_path(path, ".npz")
        np.savez_compressed(tmp_path, **arrays)
        return _copy_tmp_to_target(tmp_path, path)
    np.savez_compressed(path, **arrays)
    return path


# dbfs arrays use temp files
def save_npy(path: str | Path, array) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if uses_dbfs_fuse(path):
        tmp_path = _tmp_write_path(path, ".npy")
        np.save(tmp_path, array)
        return _copy_tmp_to_target(tmp_path, path)
    np.save(path, array)
    return path


def airxiv_figure_subdir(chart_variant: str, override: str = "") -> str:
    chart_variant = (chart_variant or "").strip()
    override = (override or "").strip()
    if override:
        return override
    if chart_variant in {"", "original", "legacy"}:
        return "arxiv_operator_evolution"
    return f"arxiv_operator_evolution_{chart_variant}"


def grokking_figure_subdir(chart_variant: str) -> str:
    chart_variant = (chart_variant or "").strip()
    if chart_variant in {"", "original", "legacy"}:
        return "grokking"
    return f"grokking_{chart_variant}"


# --- Runtime classes and configuration

# path/config containers that stop every notebook re declaring the same paths
@dataclass(frozen=True)
class AirXivRuntime:
    root: Path
    chart_variant: str
    figure_subdir: str
    figure_dir: Path
    output_dir: Path
    operator_dir: Path
    qwen3_embeddings_dir: Path
    specter2_embeddings_dir: Path
    qwen3_gaga_dir: Path
    specter2_gaga_dir: Path
    corpus_dir: Path


@dataclass(frozen=True)
class GrokkingRuntime:
    root: Path
    chart_variant: str
    figure_subdir: str
    figure_dir: Path
    activation_dir: Path

    def result_dir(self, name: str) -> Path:
        return ensure_dir(self.root / "results" / name)


# this is where the airxiv folder defaults and overrides are set
# ensure_dir creates missing folders before the notebooks write outputs
def configure_airxiv_runtime(
    chart_variant: str | None = None,
    fig_subdir_override: str | None = None,
) -> AirXivRuntime:
    root = thesis_root()
    chart_variant = (
        chart_variant
        if chart_variant is not None
        else notebook_param("AIRXIV_CHART_VARIANT", "new_charts")
    ).strip()
    fig_subdir = airxiv_figure_subdir(
        chart_variant,
        fig_subdir_override
        if fig_subdir_override is not None
        else notebook_param("AIRXIV_FIG_SUBDIR", ""),
    )
    default_output_dir = root / "results" / "arxiv_pipeline"
    default_operator_dir = root / "results" / "arxiv_operator_evolution"

    output_dir = Path(notebook_param("AIRXIV_OUTPUT_DIR", str(default_output_dir)))
    operator_dir = Path(notebook_param("AIRXIV_OPERATOR_DIR", str(default_operator_dir)))
    figure_dir = Path(notebook_param("AIRXIV_FIGURE_DIR", str(root / "figures" / fig_subdir)))
    qwen3_embeddings_dir = Path(
        notebook_param("AIRXIV_QWEN3_EMBEDDINGS_DIR", str(root / "qwen3_embeddings"))
    )
    specter2_embeddings_dir = Path(
        notebook_param("AIRXIV_SPECTER2_EMBEDDINGS_DIR", str(root / "specter2_embeddings"))
    )
    qwen3_gaga_dir = Path(notebook_param("AIRXIV_QWEN3_GAGA_DIR", str(operator_dir)))
    specter2_gaga_dir = Path(
        notebook_param("AIRXIV_SPECTER2_GAGA_DIR", str(root / "results" / "specter2_gaga"))
    )
    corpus_dir = Path(notebook_param("AIRXIV_CORPUS_DIR", str(root / "corpus")))

    return AirXivRuntime(
        root=root,
        chart_variant=chart_variant,
        figure_subdir=fig_subdir,
        figure_dir=ensure_dir(figure_dir),
        output_dir=ensure_dir(output_dir),
        operator_dir=ensure_dir(operator_dir),
        qwen3_embeddings_dir=ensure_dir(qwen3_embeddings_dir),
        specter2_embeddings_dir=ensure_dir(specter2_embeddings_dir),
        qwen3_gaga_dir=ensure_dir(qwen3_gaga_dir),
        specter2_gaga_dir=ensure_dir(specter2_gaga_dir),
        corpus_dir=ensure_dir(corpus_dir),
    )


# grokking uses the same layout runtime as airxiv but the activation folder can be swapped to make the analysis robust over seeds
def configure_grokking_runtime(chart_variant: str | None = None) -> GrokkingRuntime:
    root = thesis_root()
    chart_variant = (
        chart_variant
        if chart_variant is not None
        else notebook_param("GROKKING_CHART_VARIANT", "new_charts")
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
    ax.axvspan(
        GROKKING_START,
        GROKKING_END,
        color=GROKKING_SHADE,
        alpha=GROKKING_ALPHA,
        zorder=0,
        label="grokking window" if label else None,
    )


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


def project_spd(matrix: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(matrix)
    evals = np.maximum(evals, floor)
    projected = evecs @ np.diag(evals) @ evecs.T
    return 0.5 * (projected + projected.T)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"rho": None, "p_value": None}
    stat = spearmanr(x, y)
    return {"rho": float(stat.statistic), "p_value": float(stat.pvalue)}


def load_l0(path: str | Path) -> np.ndarray:
    path = Path(path)
    # some older cached operator files were sparse save_npz files
    # and newer ones are dense .npz files with an l0 key
    data = np.load(path, allow_pickle=True)
    if "L0" in data.files:
        return data["L0"].astype(np.float64)
    from scipy.sparse import load_npz as sparse_load_npz

    return sparse_load_npz(path).toarray().astype(np.float64)


# --- Bures Wasserstein matrix geometry

def load_L0(path: str | Path) -> np.ndarray:
    return load_l0(path)


def real_sqrtm(a: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(a)
    evals = np.maximum(evals, 0.0)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.T


def real_inv_sqrtm(a: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(a)
    evals = np.maximum(evals, 1e-12)
    return evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T


# bw is always applied after operators are turned into covariance matrices
# outputs are symmetrised to avoid small numerical asymmetries
def bw_distance(s0: np.ndarray, s1: np.ndarray) -> float:
    sqrt_s0 = real_sqrtm(s0)
    inner = sqrt_s0 @ s1 @ sqrt_s0
    sqrt_inner = real_sqrtm(inner)
    d_sq = np.trace(s0) + np.trace(s1) - 2.0 * np.trace(sqrt_inner)
    return float(np.sqrt(max(d_sq, 0.0)))


def linear_covariance_interpolation(s0: np.ndarray, s1: np.ndarray, tau: float) -> np.ndarray:
    interpolated = (1.0 - tau) * s0 + tau * s1
    return 0.5 * (interpolated + interpolated.T)


def bw_optimal_map(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    sqrt_s0 = real_sqrtm(s0)
    inv_sqrt_s0 = real_inv_sqrtm(s0)
    inner = sqrt_s0 @ s1 @ sqrt_s0
    return inv_sqrt_s0 @ real_sqrtm(inner) @ inv_sqrt_s0


def bw_geodesic(s0: np.ndarray, s1: np.ndarray, tau: float) -> np.ndarray:
    transport = bw_optimal_map(s0, s1)
    interpolant = (1.0 - tau) * np.eye(s0.shape[0]) + tau * transport
    geodesic = interpolant @ s0 @ interpolant.T
    return 0.5 * (geodesic + geodesic.T)


def bw_geodesic_from_map(s0: np.ndarray, transport: np.ndarray, tau: float) -> np.ndarray:
    interpolant = (1.0 - tau) * np.eye(s0.shape[0]) + tau * transport
    geodesic = interpolant @ s0 @ interpolant.T
    return 0.5 * (geodesic + geodesic.T)


# this tangent is what we decompose into drift and mixing
def bw_tangent(s0: np.ndarray, s1: np.ndarray) -> np.ndarray:
    sqrt_s0 = real_sqrtm(s0)
    inv_sqrt_s0 = real_inv_sqrtm(s0)
    inner = sqrt_s0 @ s1 @ sqrt_s0
    t_star = inv_sqrt_s0 @ real_sqrtm(inner) @ inv_sqrt_s0
    tangent = (t_star - np.eye(s0.shape[0])) @ s0 + s0 @ (t_star - np.eye(s0.shape[0])).T
    return 0.5 * (tangent + tangent.T)


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


def eigenvectors_to_array(raw_vectors, n_expected: int) -> np.ndarray | None:
    if raw_vectors is None:
        return None
    if isinstance(raw_vectors, np.ndarray) and raw_vectors.dtype != object:
        arr = np.real(raw_vectors)
        return arr if arr.ndim == 2 else None

    for attr in ("coeffs", "array", "coefficients", "values"):
        if hasattr(raw_vectors, attr):
            value = getattr(raw_vectors, attr)
            value = value() if callable(value) else value
            arr = np.asarray(value)
            if arr.dtype != object and arr.ndim == 2:
                arr = np.real(arr)
                batch_shape = getattr(raw_vectors, "batch_shape", ())
                if batch_shape and int(batch_shape[0]) == arr.shape[0]:
                    return arr.T
                return arr

    vectors = []
    n_vectors = min(len(raw_vectors), n_expected) if hasattr(raw_vectors, "__len__") else 0

    for idx in range(n_vectors):
        vector = raw_vectors[idx]
        for attr in ("array", "coefficients", "coeffs", "values"):
            if hasattr(vector, attr):
                value = getattr(vector, attr)
                vector = value() if callable(value) else value
                break
        arr = np.asarray(vector)
        if arr.dtype == object:
            continue
        vectors.append(np.real(arr).reshape(-1))
    return np.column_stack(vectors) if vectors else None


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


# --- Plot scaling and covariance ellipses

def tail_zoom_upper(series_list, start_idx: int = 1, floor: float = 0.1, pad: float = 0.12) -> float:
    maxima = []
    for series in series_list:
        arr = np.asarray(series, dtype=float)
        if arr.size == 0:
            continue
        tail = arr[start_idx:] if arr.size > start_idx else arr
        tail = tail[np.isfinite(tail)]
        if tail.size:
            maxima.append(float(np.max(tail)))
    if not maxima:
        return floor
    return max(floor, max(maxima) * (1.0 + pad))


def normalised_laplacian_covariance_in_reference_basis(
    x_red: np.ndarray,
    ref_basis: np.ndarray,
    k: int,
    knn: int = 15,
    eps: float = 0.01,
) -> np.ndarray:
    # grokking version of 'turn laplacian spectrum into a covariance/resolvent object before bw'
    laplacian = build_normalised_laplacian(x_red, k=knn)
    evals, evecs = eigh(laplacian)
    evals = np.maximum(evals, 0.0)
    cov_evals = 1.0 / (evals[1 : k + 1] + eps)
    phi = evecs[:, 1 : k + 1]
    change_of_basis = ref_basis.T @ phi
    sigma = change_of_basis @ np.diag(cov_evals) @ change_of_basis.T
    return 0.5 * (sigma + sigma.T)


def subspace_bw_distance(
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
    idx_i: int,
    idx_j: int,
) -> float:
    idx = [idx_i, idx_j]
    return bw_distance(sigma_a[np.ix_(idx, idx)], sigma_b[np.ix_(idx, idx)])


def pair_marginal_series(all_cov_mats: dict, available: list[int], idx_i: int, idx_j: int) -> dict[str, np.ndarray]:
    eig_major = []
    eig_minor = []
    circularity = []
    for ep in available:
        sub = all_cov_mats[ep][np.ix_([idx_i, idx_j], [idx_i, idx_j])]
        evals = np.linalg.eigvalsh(sub)
        evals = np.maximum(evals, 0.0)
        eig_minor.append(evals[0])
        eig_major.append(evals[1])
        circularity.append(evals[0] / max(evals[1], 1e-12))
    return {
        "major": np.asarray(eig_major, dtype=float),
        "minor": np.asarray(eig_minor, dtype=float),
        "circ": np.asarray(circularity, dtype=float),
    }

def plot_ellipse_overlay(
    ax,
    all_cov_mats: dict,
    idx_i: int,
    idx_j: int,
    epochs: list[int],
    title: str,
    xlabel: str,
    ylabel: str,
    cmap,
) -> list[Line2D]:
    colors = cmap(np.linspace(0.15, 0.9, len(epochs)))
    handles = []
    max_extent = 0.0

    for color, ep in zip(colors, epochs):
        if ep not in all_cov_mats:
            continue
        sub = all_cov_mats[ep][np.ix_([idx_i, idx_j], [idx_i, idx_j])]
        width, height, angle, _ = ellipse_from_covariance(sub)
        max_extent = max(max_extent, width, height)
        ellipse = Ellipse(
            xy=(0, 0),
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor=color,
            linewidth=2.2,
            alpha=0.95,
        )
        ax.add_patch(ellipse)
        handles.append(Line2D([0], [0], color=color, lw=2.2, label=f"{ep}"))

    pad = max(max_extent * 0.8, 0.8)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.7", linewidth=0.8, alpha=0.45, zorder=0)
    ax.axvline(0, color="0.7", linewidth=0.8, alpha=0.45, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.12)
    return handles


# --- AirXiv temporal and transport diagnostics

def resolvent_in_reference_basis_from_operator(
    operator: np.ndarray,
    ref_basis: np.ndarray,
    alpha: float = 0.1,
    k: int | None = None,
) -> np.ndarray:
    # operator is l0 but bw compares (l0 + alpha i)^(-1)
    # in a chosen basis
    n_basis = ref_basis.shape[0]
    sigma_full = np.linalg.inv(operator + alpha * np.eye(n_basis))
    sigma_full = 0.5 * (sigma_full + sigma_full.T)
    sigma_ref = ref_basis.T @ sigma_full @ ref_basis
    if k is not None:
        sigma_ref = sigma_ref[:k, :k]
    return 0.5 * (sigma_ref + sigma_ref.T)


def temporal_signal_triplet(
    x_src: np.ndarray,
    x_hold: np.ndarray,
    x_tgt: np.ndarray,
    n_interp: int = 300,
    k: int = 20,
    seed: int = 42,
) -> dict[str, float | int]:
    local_rng = np.random.default_rng(seed)
    n_src, n_hold, n_tgt = len(x_src), len(x_hold), len(x_tgt)
    x_pool = np.vstack([x_src, x_hold, x_tgt])
    labels = np.array([0] * n_src + [1] * n_hold + [2] * n_tgt)

    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(x_pool)

    idx_src = local_rng.choice(n_src, n_interp, replace=True)
    idx_tgt = local_rng.choice(n_tgt, n_interp, replace=True)
    x_mid = 0.5 * x_src[idx_src] + 0.5 * x_tgt[idx_tgt]

    _, indices = nn.kneighbors(x_mid)
    neighbour_labels = labels[indices]
    holdout_counts = (neighbour_labels == 1).sum(axis=1)
    holdout_frac = float(holdout_counts.mean() / k)
    nn1_holdout_frac = float((neighbour_labels[:, 0] == 1).mean())
    baseline_frac = n_hold / (n_src + n_hold + n_tgt)

    return {
        "holdout_frac": holdout_frac,
        "baseline_frac": float(baseline_frac),
        "nn1_holdout_frac": nn1_holdout_frac,
        "enrichment": holdout_frac / max(baseline_frac, 1e-10),
        "n_src": n_src,
        "n_hold": n_hold,
        "n_tgt": n_tgt,
    }


# --- Tangent decomposition

# instructive function includes near degenerate mode, gw matching and bw decomp of resolvent
def bw_generator_decomposition(
    phi_a: np.ndarray,
    lambda_a: np.ndarray,
    phi_b: np.ndarray,
    lambda_b: np.ndarray,
    x_a: np.ndarray,
    x_b: np.ndarray,
    k_use: int = 20,
    alpha: float = 0.1,
    degenerate_threshold: float = 1e-3,
    gw_reg: float = 0.01,
    gw_max_iter: int = 200,
) -> dict[str, float]:
    import ot

    # corrected resolvent tangent split
    k_use = min(k_use, len(lambda_a), len(lambda_b), phi_a.shape[1], phi_b.shape[1])
    phi_a_k, lambda_a_k = phi_a[:, :k_use], lambda_a[:k_use]
    phi_b_k, lambda_b_k = phi_b[:, :k_use], lambda_b[:k_use]
    n_a, n_b = x_a.shape[0], x_b.shape[0]

    c_a = ot.dist(x_a, x_a, metric="sqeuclidean")
    c_b = ot.dist(x_b, x_b, metric="sqeuclidean")
    c_a /= c_a.max() + 1e-10
    c_b /= c_b.max() + 1e-10
    a = np.ones(n_a) / n_a
    b = np.ones(n_b) / n_b

    pi = ot.gromov.entropic_gromov_wasserstein(
        c_a,
        c_b,
        a,
        b,
        loss_fun="square_loss",
        epsilon=gw_reg,
        max_iter=gw_max_iter,
        verbose=False,
    )

    alignment = phi_a_k.T @ (pi * n_a) @ phi_b_k
    sigma_a = 1.0 / (np.clip(lambda_a_k, 1e-8, None) + alpha)
    sigma_b = 1.0 / (np.clip(lambda_b_k, 1e-8, None) + alpha)
    sigma = np.diag(sigma_a)
    sigma_b_aligned = alignment @ np.diag(sigma_b) @ alignment.T
    sigma_b_aligned = 0.5 * (sigma_b_aligned + sigma_b_aligned.T)
    tangent = bw_tangent(sigma, sigma_b_aligned)

    # diagonal means drift
    drift_energy = float(np.sum(np.diag(tangent) ** 2))
    # off diagonal means mixing
    mixing_energy = 0.0
    for i in range(k_use):
        for j in range(i + 1, k_use):
            s_ij = tangent[i, j]
            if abs(s_ij) < 1e-15:
                continue
            gap = abs(sigma_a[i] - sigma_a[j])
            scale = max(abs(sigma_a[i]), abs(sigma_a[j]), 1e-15)
            # skip near degenerate pairs
            if gap / scale >= degenerate_threshold:
                mixing_energy += 2.0 * s_ij**2

    total_energy = float(np.linalg.norm(tangent, "fro") ** 2)
    # fractions may not sum
    denom = max(total_energy, 1e-15)
    return {
        "drift_frac": drift_energy / denom,
        "rotation_frac": mixing_energy / denom,
        "total_energy": total_energy,
    }


def spectral_tangent_decomposition(s0: np.ndarray, s1: np.ndarray) -> dict[str, float]:
    tangent = bw_tangent(s0, s1)
    diagonal = np.diag(np.diag(tangent))
    off_diagonal = tangent - diagonal
    drift_energy = float(np.linalg.norm(diagonal, "fro") ** 2)
    mixing_energy = float(np.linalg.norm(off_diagonal, "fro") ** 2)
    total = drift_energy + mixing_energy
    return {
        "drift_frac": drift_energy / max(total, 1e-15),
        "mixing_frac": mixing_energy / max(total, 1e-15),
        "bw_distance": bw_distance(s0, s1),
        "tangent_norm": float(np.sqrt(total)),
    }


def source_basis_bw_tangent_split(
    phi_a: np.ndarray,
    lambda_a: np.ndarray,
    phi_b: np.ndarray,
    lambda_b: np.ndarray,
    eps: float = 0.01,
    psd_floor: float = 1e-12,
) -> tuple[float, float]:
    k_use = min(len(lambda_a), len(lambda_b), phi_a.shape[1], phi_b.shape[1])
    phi_a = phi_a[:, :k_use]
    phi_b = phi_b[:, :k_use]
    lambda_a = lambda_a[:k_use]
    lambda_b = lambda_b[:k_use]

    gram = phi_a.T @ phi_b
    sigma_a = 1.0 / (lambda_a + eps)
    sigma_b = 1.0 / (lambda_b + eps)
    cov_a = np.diag(sigma_a)
    cov_b = gram @ np.diag(sigma_b) @ gram.T
    cov_b = project_spd(cov_b, floor=psd_floor)

    tangent = bw_tangent(cov_a, cov_b)
    diagonal = np.diag(np.diag(tangent))
    off_diagonal = tangent - diagonal
    drift_energy = float(np.linalg.norm(diagonal, "fro") ** 2)
    mixing_energy = float(np.linalg.norm(off_diagonal, "fro") ** 2)
    total = max(drift_energy + mixing_energy, 1e-15)
    return drift_energy / total, mixing_energy / total


# --- Toy rotations

def plane_rotation(dim: int, i: int, j: int, theta: float) -> np.ndarray:
    rotation = np.eye(dim)
    c = np.cos(theta)
    s = np.sin(theta)
    rotation[i, i] = c
    rotation[j, j] = c
    rotation[i, j] = -s
    rotation[j, i] = s
    return rotation


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


# --- Ellipse drawing and sampling

def ellipse_from_covariance(cov_2d: np.ndarray) -> tuple[float, float, float, float]:
    evals, evecs = np.linalg.eigh(cov_2d)
    evals = np.maximum(evals, 1e-12)
    width = 2.0 * np.sqrt(evals[1])
    height = 2.0 * np.sqrt(evals[0])
    angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
    circularity = float(evals[0] / evals[1])
    return float(width), float(height), float(angle), circularity


def draw_covariance_ellipse(
    ax,
    cov_2d: np.ndarray,
    color: str = "blue",
    alpha: float = 0.75,
    label: str | None = None,
    linewidth: float = 1.4,
    sigma_scale: float = 2.0,
) -> float:
    width, height, angle, _ = ellipse_from_covariance(cov_2d)
    width *= sigma_scale
    height *= sigma_scale
    ellipse = Ellipse(
        xy=(0, 0),
        width=width,
        height=height,
        angle=angle,
        edgecolor=color,
        facecolor="none",
        alpha=alpha,
        linewidth=linewidth,
        label=label,
    )
    ax.add_patch(ellipse)
    return max(width, height)


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


# --- Hodge decomposition helpers

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
