import runpy
import unittest
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parents[1]


class ScriptImportTests(unittest.TestCase):
    def test_analysis_pipeline_and_validation_scripts_are_import_safe(self):
        paths = [
            *sorted((REPRO_ROOT / "Grokking" / "Analysis").glob("*.py")),
            *sorted((REPRO_ROOT / "Grokking" / "Pipelines").glob("*.py")),
            *sorted((REPRO_ROOT / "Grokking" / "Validation").glob("*.py")),
        ]
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                runpy.run_path(str(path), run_name="import_check")


if __name__ == "__main__":
    unittest.main()
