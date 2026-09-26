import unittest
from pathlib import Path

from ml_sherlock.config import SherlockConfig
from ml_sherlock.models.trainer import BaselineTrainer


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
                config = SherlockConfig.from_yaml(root / "examples" / name / "sherlock.yaml")
                self.assertEqual(config.data.target, target)
                self.assertEqual(config.models.candidates, list(BaselineTrainer.SUPPORTED_MODELS))
                self.assertEqual(config.data.train.name, "train.csv")
                self.assertEqual(config.data.production.name, "production.csv")
                expected_db = root / "examples" / name / "mlflow.db"
                self.assertEqual(config.tracking.uri, f"sqlite:///{expected_db.as_posix()}")


if __name__ == "__main__":
    unittest.main()
