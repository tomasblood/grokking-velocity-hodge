import unittest

from grokking_velocity_hodge.config import ExperimentConfig
from grokking_velocity_hodge.robustness import (
    HodgeSweepConfig,
    one_at_a_time_settings,
    representative_pairs,
)


class RobustnessConfigurationTests(unittest.TestCase):
    def test_one_at_a_time_grid_contains_baseline_without_full_factorial(self):
        experiment = ExperimentConfig()
        sweep = HodgeSweepConfig()
        settings = one_at_a_time_settings(experiment, sweep)
        self.assertEqual(len(settings), 7)
        self.assertIn(
            {"pca_dim": 10, "knn": 15, "n_basis": 50},
            settings,
        )

    def test_representative_pairs_cover_all_training_phases(self):
        pairs = representative_pairs(list(range(0, 5001, 500)), 1500, 4000, pairs_per_phase=2)
        self.assertEqual({pair["phase"] for pair in pairs}, {"pre", "transition", "post"})
        self.assertLessEqual(sum(pair["phase"] == "pre" for pair in pairs), 2)
        self.assertLessEqual(sum(pair["phase"] == "transition" for pair in pairs), 2)
        self.assertLessEqual(sum(pair["phase"] == "post" for pair in pairs), 2)


if __name__ == "__main__":
    unittest.main()
