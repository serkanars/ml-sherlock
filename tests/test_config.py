from pathlib import Path
import tempfile
import unittest

from autoresearch.config import InvestigationConfig
from autoresearch.models.trainer import BaselineTrainer


class InvestigationConfigTests(unittest.TestCase):
    def write_config(self, body):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        config = Path(directory.name) / "sherlock.yaml"
        config.write_text(body, encoding="utf-8")
        return config

    def test_default_model_candidates_include_all_supported_models(self):
        config = self.write_config(
            "project:\n  target: y\ndata:\n  train_path: train.csv\n  production_path: prod.csv\n"
        )
        loaded = InvestigationConfig.from_yaml(config)
        self.assertEqual(loaded.model_candidates, BaselineTrainer.SUPPORTED_MODELS)

    def test_unknown_model_candidate_is_rejected(self):
        config = self.write_config(
            "project:\n  target: y\ndata:\n  train_path: train.csv\n  production_path: prod.csv\n"
            "model:\n  candidates: [xgboost, unknown_booster]\n"
        )
        with self.assertRaisesRegex(ValueError, "unknown_booster"):
            InvestigationConfig.from_yaml(config)
