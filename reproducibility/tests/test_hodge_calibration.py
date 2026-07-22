import unittest

from grokking_velocity_hodge.calibration import run_synthetic_hodge_calibration


class RealHodgeCalibrationTests(unittest.TestCase):
    def test_exact_coexact_and_harmonic_fields(self):
        result = run_synthetic_hodge_calibration()
        self.assertTrue(result["passed"], result["fields"])


if __name__ == "__main__":
    unittest.main()
