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

    def test_enabled_segments_require_columns(self):
        path = self.write_config(
            "version: 1\ndata:\n  target: y\n  train: train.csv\n  production: prod.csv\n"
            "investigation:\n  segments:\n    enabled: true\n"
        )
        with self.assertRaisesRegex(ValidationError, "requires at least one column"):
            SherlockConfig.from_yaml(path)

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
