import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from grokking_velocity_hodge.config import ExperimentConfig
from grokking_velocity_hodge.provenance import compare_bw_files
from grokking_velocity_hodge.runtime import (
    build_normalised_laplacian,
    bw_distance,
    covariance_in_reference_basis,
    dimmed_phase_cmap,
    edge_padded_moving_average,
    heat_kernel_in_reference_basis,
    hodge_decompose_velocity,
    laplacian_spectrum,
)
from grokking_velocity_hodge.seed_sweep import (
    load_seed_sweep_config,
    summarise_hodge,
    summarise_resolvent_bw,
)
from grokking_velocity_hodge.summary import effective_dimension, mean_sd, summarise_transition_series


class RuntimeTests(unittest.TestCase):
    def test_environment_configuration_and_metadata_epochs(self):
        with patch.dict(
            "os.environ",
            {"GROKKING_N_EPOCHS": "1000", "GROKKING_SAVE_EVERY": "250", "GROKKING_KNN": "9"},
            clear=False,
        ):
            config = ExperimentConfig.from_environment()
        self.assertEqual(config.epochs, [0, 250, 500, 750, 1000])
        self.assertEqual(config.knn, 9)
        self.assertEqual(config.checkpoint_epochs({"saved_epochs": [0, 400, 800]}), [0, 400, 800])

    def test_seed_sweep_paths_resolve_from_repository_root(self):
        config_path = Path(__file__).resolve().parents[1] / "Grokking" / "config" / "seed_sweep.json"
        config = load_seed_sweep_config(config_path)
        for root in config["roots"].values():
            self.assertTrue(Path(root).is_absolute())

    def test_edge_padded_moving_average(self):
        actual = edge_padded_moving_average(np.array([1.0, 2.0, 3.0]), window=3)
        np.testing.assert_allclose(actual, [4.0 / 3.0, 2.0, 8.0 / 3.0])

    def test_bw_distance_for_commuting_diagonal_matrices(self):
        sigma_0 = np.diag([1.0, 4.0])
        sigma_1 = np.diag([4.0, 9.0])
        self.assertAlmostEqual(bw_distance(sigma_0, sigma_1), np.sqrt(2.0))

    def test_graph_laplacian_is_symmetric_and_has_valid_spectrum(self):
        points = np.array([[0.0], [1.0], [2.0], [4.0]])
        laplacian = build_normalised_laplacian(points, k=2)
        np.testing.assert_allclose(laplacian, laplacian.T)
        eigenvalues, eigenvectors = laplacian_spectrum(laplacian, k=2)
        self.assertEqual(eigenvectors.shape, (4, 2))
        self.assertTrue(np.all(eigenvalues >= 0.0))
        self.assertTrue(np.all(eigenvalues <= 2.0 + 1e-12))

    def test_operator_embeddings_preserve_diagonal_case(self):
        eigenvalues = np.array([0.1, 0.2])
        basis = np.eye(3)[:, :2]
        covariance = covariance_in_reference_basis(eigenvalues, basis, basis, eps=0.01, floor=0.0)
        heat = heat_kernel_in_reference_basis(eigenvalues, basis, basis, tau=1.0, floor=0.0)
        np.testing.assert_allclose(covariance, np.diag(1.0 / (eigenvalues + 0.01)))
        np.testing.assert_allclose(heat, np.diag(np.exp(-eigenvalues)))

    def test_dimmed_phase_colormap_is_stable(self):
        cmap = dimmed_phase_cmap("hsv", 0.58, 0.92)
        self.assertEqual(cmap.name, "hsv_dimmed")
        self.assertEqual(cmap.N, 256)

    def test_hodge_wrapper_preserves_component_energy_fractions(self):
        class Form:
            def __init__(self, norm):
                self._norm = norm

            def norm(self):
                return self._norm

            def d(self):
                return self

            def codifferential(self):
                return self

        class Omega(Form):
            def hodge_decomposition(self):
                return Form(2.0), Form(1.0), Form(1.0)

        class Model:
            def form(self, velocity, degree):
                self.velocity = velocity
                self.degree = degree
                return Omega(2.0)

        class DiffusionGeometry:
            @staticmethod
            def from_point_cloud(points, knn_kernel, n_function_basis):
                return Model()

        class DGModule:
            pass

        DGModule.DiffusionGeometry = DiffusionGeometry

        class PCA:
            def __init__(self, n_components, svd_solver):
                self.n_components = n_components

            def fit_transform(self, values):
                self.components_ = np.eye(values.shape[1])[: self.n_components]
                return values[:, : self.n_components]

        points = np.arange(12.0).reshape(4, 3)
        result = hodge_decompose_velocity(points, np.ones_like(points), PCA, DGModule, pca_dim=2)
        self.assertAlmostEqual(result["exact"], 4.0 / 6.0)
        self.assertAlmostEqual(result["coexact"], 1.0 / 6.0)
        self.assertAlmostEqual(result["harmonic"], 1.0 / 6.0)
        self.assertEqual(result["total_energy"], 4.0)


class SummaryTests(unittest.TestCase):
    def test_effective_dimension(self):
        points = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        self.assertAlmostEqual(effective_dimension(points), 2.0)

    def test_transition_summary_schema_and_values(self):
        summary = summarise_transition_series(
            [250, 1750, 2250, 4250, 4750],
            [10.0, 6.0, 8.0, 2.0, 4.0],
        )
        self.assertEqual(summary["initial_0_500"], 10.0)
        self.assertEqual(summary["transition_peak"], 8.0)
        self.assertEqual(summary["transition_mean"], 7.0)
        self.assertEqual(summary["post_mean"], 3.0)
        self.assertAlmostEqual(summary["transition_peak_over_post_mean"], 8.0 / 3.0)

    def test_mean_sd_ignores_missing_values_when_requested(self):
        self.assertEqual(
            mean_sd([1.0, None, 3.0], ignore_none=True, include_n=True),
            {
                "mean": 2.0,
                "sd": np.sqrt(2.0),
                "n": 2,
            },
        )

    def test_hodge_seed_summary_uses_transition_window(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results" / "grokking_dg_velocity_hodge"
            output.mkdir(parents=True)
            payload = {
                "config": {"pca_dim": 10, "knn": 15, "n_basis": 50},
                "pairs": [
                    {"midpoint": 1000, "exact": 0.6, "coexact": 0.3, "harmonic": 0.1},
                    {"midpoint": 2000, "exact": 0.3, "coexact": 0.6, "harmonic": 0.1},
                    {"midpoint": 3500, "exact": 0.5, "coexact": 0.4, "harmonic": 0.1},
                    {"midpoint": 5000, "exact": 0.7, "coexact": 0.2, "harmonic": 0.1},
                ],
            }
            (output / "velocity_hodge.json").write_text(json.dumps(payload), encoding="utf-8")
            summary = summarise_hodge(directory)

        self.assertTrue(summary["available"])
        self.assertAlmostEqual(summary["transition_mean_exact"], 0.4)
        self.assertAlmostEqual(summary["transition_mean_coexact"], 0.5)
        self.assertAlmostEqual(summary["transition_mean_coexact_minus_exact"], 0.1)

    def test_resolvent_seed_summary_preserves_consecutive_bw_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results" / "grokking_resolvent_bw"
            output.mkdir(parents=True)
            payload = {
                "bw_distances_consecutive": {
                    "midpoint_epochs": [250, 1750, 2250, 4250, 4750],
                    "distances": [10.0, 6.0, 8.0, 2.0, 4.0],
                }
            }
            (output / "resolvent_bw_results.json").write_text(json.dumps(payload), encoding="utf-8")
            summary = summarise_resolvent_bw(directory)

        self.assertTrue(summary["available"])
        self.assertEqual(summary["summary"]["transition_peak"], 8.0)
        self.assertEqual(summary["summary"]["post_mean"], 3.0)

    def test_bw_provenance_comparison_distinguishes_exactness_from_stability(self):
        with tempfile.TemporaryDirectory() as directory:
            cached_path = Path(directory) / "cached.json"
            rerun_path = Path(directory) / "rerun.json"
            midpoint_epochs = [250, 1750, 2250, 4250, 4750]
            cached = {
                "bw_distances_consecutive": {
                    "midpoint_epochs": midpoint_epochs,
                    "distances": [10.0, 6.0, 8.0, 2.0, 4.0],
                }
            }
            rerun = {
                "bw_distances_consecutive": {
                    "midpoint_epochs": midpoint_epochs,
                    "distances": [10.0, 6.0, 8.01, 2.0, 4.0],
                }
            }
            cached_path.write_text(json.dumps(cached), encoding="utf-8")
            rerun_path.write_text(json.dumps(rerun), encoding="utf-8")
            report = compare_bw_files(cached_path, rerun_path)

        self.assertFalse(report["series_exactly_equal"])
        self.assertTrue(report["conclusion_stable"])


if __name__ == "__main__":
    unittest.main()
