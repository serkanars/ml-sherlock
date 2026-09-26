import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ml_sherlock import Sherlock


class SherlockPublicApiTests(unittest.TestCase):
    def test_public_import_exposes_sherlock(self):
        from ml_sherlock import Sherlock as ImportedSherlock

        self.assertIs(ImportedSherlock, Sherlock)

    def test_investigate_runs_complete_config_workflow(self):
        expected = {"fit": {"run_id": "baseline"}, "investigation": {"report": "report.html"}}
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "sherlock.yaml"
            config.write_text(
                "version: 1\ndata:\n  target: y\n  train: train.csv\n  production: prod.csv\n",
                encoding="utf-8",
            )
            with patch("ml_sherlock.sherlock.ResearchRunner.run_config", return_value=expected) as run:
                sherlock = Sherlock(config=config)
                result = sherlock.investigate()

        run.assert_called_once_with(sherlock.config)
        self.assertIs(result, expected)
        self.assertIs(sherlock.result, expected)
        self.assertEqual(sherlock.fit_result, expected["fit"])
        self.assertEqual(sherlock.investigation_result, expected["investigation"])

    def test_missing_config_has_actionable_error(self):
        missing = Path("missing-sherlock-config.yaml")
        with self.assertRaisesRegex(FileNotFoundError, "Sherlock config not found"):
            Sherlock(config=missing)


if __name__ == "__main__":
    unittest.main()
