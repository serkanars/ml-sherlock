import unittest
from pathlib import Path

from autoresearch.config import InvestigationConfig
from autoresearch.models.trainer import BaselineTrainer


class RealDataExampleTests(unittest.TestCase):
    def test_all_real_data_examples_have_valid_configs(self):
        root = Path(__file__).resolve().parents[1]
        expected_targets = {
            "nyc_taxi": "trip_duration_minutes",
            "citi_bike": "ride_duration_minutes",
            "seoul_bike": "rented_bike_count",
            "california_housing": "median_house_value",
        }
        for name, target in expected_targets.items():
            with self.subTest(example=name):
                config = InvestigationConfig.from_yaml(root / "examples" / name / "sherlock.yaml")
                self.assertEqual(config.target, target)
                self.assertEqual(config.model_candidates, BaselineTrainer.SUPPORTED_MODELS)
                self.assertEqual(config.train_path.name, "train.csv")
                self.assertEqual(config.production_path.name, "production.csv")


if __name__ == "__main__":
    unittest.main()
