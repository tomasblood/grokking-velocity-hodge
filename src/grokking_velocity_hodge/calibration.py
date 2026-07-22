"""Synthetic end-to-end checks for the DiffusionGeometry Hodge pipeline."""

import diffusion_geometry as dg
import numpy as np
from sklearn.decomposition import PCA

from .runtime import hodge_decompose_velocity


def _planar_grid(size: int = 5) -> np.ndarray:
    coordinates = np.linspace(-1.0, 1.0, size)
    return np.asarray([[x, y, 0.0] for x in coordinates for y in coordinates])


def _annulus(rings: int = 3, angles: int = 24) -> np.ndarray:
    radii = np.linspace(0.6, 1.0, rings)
    theta = np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False)
    return np.asarray([[r * np.cos(t), r * np.sin(t), 0.0] for r in radii for t in theta])


def _discrete_harmonic_velocity(
    points: np.ndarray,
    *,
    knn: int = 8,
    n_basis: int = 30,
) -> np.ndarray:
    """Construct a ground-truth harmonic field in the fitted discrete complex."""
    pca = PCA(n_components=2, svd_solver="full")
    reduced = pca.fit_transform(points)
    radius_sq = np.sum(reduced**2, axis=1)
    angular_velocity = np.column_stack((-reduced[:, 1] / radius_sq, reduced[:, 0] / radius_sq))
    model = dg.DiffusionGeometry.from_point_cloud(
        reduced.astype(np.float64),
        knn_kernel=knn,
        n_function_basis=n_basis,
    )
    form = model.form(angular_velocity.astype(np.float64), degree=1)
    _, _, harmonic = form.hodge_decomposition()
    return harmonic.to_ambient() @ pca.components_


def run_synthetic_hodge_calibration() -> dict:
    grid = _planar_grid()
    radial = grid.copy()
    rotational = np.column_stack((-grid[:, 1], grid[:, 0], np.zeros(len(grid))))

    common = {"pca_dim": 2, "pca_solver": "full", "knn": 8, "n_basis": 20}
    exact = hodge_decompose_velocity(grid, radial, PCA, dg, **common)
    coexact = hodge_decompose_velocity(grid, rotational, PCA, dg, **common)

    annulus = _annulus()
    harmonic_velocity = _discrete_harmonic_velocity(annulus)
    harmonic = hodge_decompose_velocity(
        annulus,
        harmonic_velocity,
        PCA,
        dg,
        pca_dim=2,
        pca_solver="full",
        knn=8,
        n_basis=30,
    )

    thresholds = {"exact": 0.85, "coexact": 0.85, "harmonic": 0.80}
    checks = {
        "gradient_is_exact": exact["exact"] >= thresholds["exact"],
        "rotation_is_coexact": coexact["coexact"] >= thresholds["coexact"],
        "discrete_harmonic_is_harmonic": harmonic["harmonic"] >= thresholds["harmonic"],
    }
    return {
        "config": {
            "grid_size": 5,
            "annulus_rings": 3,
            "annulus_angles": 24,
            "knn": 8,
            "grid_basis": 20,
            "annulus_basis": 30,
        },
        "thresholds": thresholds,
        "fields": {"gradient": exact, "rotation": coexact, "harmonic": harmonic},
        "checks": checks,
        "passed": all(checks.values()),
    }
