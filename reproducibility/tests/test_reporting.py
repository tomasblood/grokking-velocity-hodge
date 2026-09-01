import json
import unittest
from pathlib import Path

from grokking_velocity_hodge.reporting import build_cross_seed_hodge_summary


class TestEightSeedReporting(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[2]
        artifact_dir = cls.root / "reproducibility" / "artifacts" / "eight_seed_hodge"
        cls.seeds = (598, 599, 777, 1001, 2027, 3141, 4096, 8080)
        cls.summary = build_cross_seed_hodge_summary(
            [
                {
                    "training_seed": seed,
                    "hodge_path": artifact_dir / f"hodge_robustness_seed{seed}.json",
                    "training_path": artifact_dir / f"training_seed{seed}.json",
                }
                for seed in cls.seeds
            ]
        )

    def test_design_and_integrity(self) -> None:
        self.assertEqual(self.summary["design"]["n_training_seeds"], 8)
        self.assertEqual(self.summary["design"]["ordinary_records"], 2016)
        self.assertEqual(self.summary["design"]["permutation_null_records"], 288)
        for run in self.summary["integrity"]:
            self.assertTrue(run["all_finite"])
            self.assertLess(run["max_fraction_sum_error"], 1e-8)

    def test_fixed_seed_level_result(self) -> None:
        fixed = self.summary["phase_conventions"]["fixed"]
        estimate = fixed["seed_level_inference"][
            "baseline_post_minus_transition_coexact_minus_exact"
        ]
        self.assertEqual(estimate["n_training_seeds"], 8)
        self.assertAlmostEqual(estimate["mean"], -0.26711775, places=7)
        self.assertAlmostEqual(estimate["interval"][0], -0.399673, places=6)
        self.assertAlmostEqual(estimate["interval"][1], -0.134562, places=6)
        self.assertEqual(
            fixed["direction_counts"]["seed_setting_shift_toward_exact"]["count"], 56
        )

    def test_event_aligned_result(self) -> None:
        event = self.summary["phase_conventions"]["event_aligned"]
        estimate = event["seed_level_inference"][
            "baseline_post_minus_transition_coexact_minus_exact"
        ]
        self.assertAlmostEqual(estimate["mean"], -0.360404, places=6)
        self.assertLess(estimate["interval"][1], 0.0)
        self.assertEqual(
            event["direction_counts"]["seed_setting_shift_toward_exact"]["count"], 56
        )

    def test_config_declares_all_runs_and_hodge_task(self) -> None:
        config_path = self.root / "reproducibility" / "Grokking" / "config" / "seed_sweep.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(tuple(run["data_seed"] for run in config["runs"]), self.seeds)
        self.assertIn("hodge_robustness", config["analysis_tasks"])


if __name__ == "__main__":
    unittest.main()
