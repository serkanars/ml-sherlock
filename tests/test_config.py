from pathlib import Path
import tempfile
import unittest
import warnings

from pydantic import ValidationError

from ml_sherlock.config import SherlockConfig
from ml_sherlock.models.trainer import BaselineTrainer


class SherlockConfigTests(unittest.TestCase):
    def write_config(self, body):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = Path(directory.name) / "sherlock.yaml"
        config.write_text(body, encoding="utf-8")
        return config

    def test_typed_defaults_include_all_supported_models_and_resolve_paths(self):
        path = self.write_config(
            "version: 1\ndata:\n  target: y\n  train: train.csv\n  production: prod.csv\n"
        )
        loaded = SherlockConfig.from_yaml(path)
        self.assertEqual(loaded.models.candidates, list(BaselineTrainer.SUPPORTED_MODELS))
        self.assertEqual(loaded.investigation.drift.alpha, 0.05)
        self.assertEqual(loaded.investigation.drift.multiple_testing, "benjamini_hochberg")
        self.assertEqual(loaded.data.train, path.parent / "train.csv")
        self.assertEqual(loaded.report.output, path.parent / "artifacts" / "report.html")
        self.assertEqual(
            loaded.tracking.uri,
            f"sqlite:///{(path.parent / 'artifacts' / 'mlflow.db').as_posix()}",
        )

    def test_unknown_model_candidate_is_rejected(self):
        path = self.write_config(
            "version: 1\ndata:\n  target: y\n  train: train.csv\n  production: prod.csv\n"
            "models:\n  candidates: [xgboost, unknown_booster]\n"
        )
        with self.assertRaisesRegex(ValidationError, "unknown_booster"):
            SherlockConfig.from_yaml(path)

    def test_unknown_keys_are_forbidden(self):
        path = self.write_config(
            "version: 1\ndata:\n  target: y\n  train: train.csv\n  production: prod.csv\n"
            "experiments:\n  max_experimnts: 10\n"
        )
        with self.assertRaisesRegex(ValidationError, "max_experimnts"):
            SherlockConfig.from_yaml(path)

    def test_enabled_segments_support_automatic_column_discovery(self):
        path = self.write_config(
            "version: 1\ndata:\n  target: y\n  train: train.csv\n  production: prod.csv\n"
            "investigation:\n  segments:\n    enabled: true\n    min_rows: 100\n"
        )
        loaded = SherlockConfig.from_yaml(path)
        self.assertTrue(loaded.investigation.segments.enabled)
        self.assertEqual(loaded.investigation.segments.columns, [])
        self.assertEqual(loaded.investigation.segments.min_rows, 100)
        self.assertEqual(loaded.investigation.segments.numeric_bins, 4)

    def test_drift_multiple_testing_policy_is_configurable(self):
        path = self.write_config(
            "version: 1\ndata:\n  target: y\n  train: train.csv\n  production: prod.csv\n"
            "investigation:\n  drift:\n    alpha: 0.1\n    multiple_testing: none\n"
        )

        loaded = SherlockConfig.from_yaml(path)

        self.assertEqual(loaded.investigation.drift.alpha, 0.1)
        self.assertEqual(loaded.investigation.drift.multiple_testing, "none")

    def test_legacy_p_value_threshold_is_accepted(self):
        path = self.write_config(
            "version: 1\ndata:\n  target: y\n  train: train.csv\n  production: prod.csv\n"
            "investigation:\n  drift:\n    p_value_threshold: 0.02\n"
        )

        loaded = SherlockConfig.from_yaml(path)

        self.assertEqual(loaded.investigation.drift.alpha, 0.02)
        self.assertEqual(loaded.investigation.drift.p_value_threshold, 0.02)

    def test_legacy_yaml_is_migrated_with_warning(self):
        path = self.write_config(
            "project:\n  target: y\ndata:\n  train_path: train.csv\n  production_path: prod.csv\n"
            "model:\n  candidates: [random_forest]\nresearch:\n  max_experiments: 3\n"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = SherlockConfig.from_yaml(path)
        self.assertEqual(loaded.data.target, "y")
        self.assertEqual(loaded.models.candidates, ["random_forest"])
        self.assertEqual(loaded.experiments.max_iterations, 3)
        self.assertTrue(any("deprecated" in str(item.message) for item in caught))
