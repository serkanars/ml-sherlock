import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from ml_sherlock.evidence import Evidence
from ml_sherlock.core.research import ResearchRunner
from ml_sherlock.monitoring import PredictionDriftAnalyzer, TargetDriftAnalyzer


class LinearModel:
    def predict(self, features):
        return features["x"].to_numpy() * 2.0 + 1.0


class DistributionDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(123)
        cls.reference_target = pd.Series(rng.normal(10, 1, 1500), name="sales")
        cls.shifted_target = pd.Series(rng.normal(13, 1, 1500), name="sales")
        cls.reference_features = pd.DataFrame({"x": rng.normal(0, 1, 1500)})
        cls.shifted_features = pd.DataFrame({"x": rng.normal(2, 1, 1500)})

    def test_target_shift_produces_target_drift_evidence(self):
        evidence = TargetDriftAnalyzer().analyze(
            self.reference_target, self.shifted_target, "sales"
        )

        self.assertIsInstance(evidence, Evidence)
        self.assertEqual(evidence.type, "target_drift")
        self.assertEqual(evidence.feature, "sales")
        self.assertTrue(evidence.metadata["drift"])
        self.assertGreater(evidence.metadata["wasserstein_distance"], 2.0)
        self.assertGreater(evidence.metadata["psi"], 0.25)
        self.assertEqual(evidence.direction, "increased")

    def test_prediction_shift_produces_prediction_drift_evidence(self):
        evidence = PredictionDriftAnalyzer().analyze(
            LinearModel(), self.reference_features, self.shifted_features
        )

        self.assertIsInstance(evidence, Evidence)
        self.assertEqual(evidence.type, "prediction_drift")
        self.assertIsNone(evidence.feature)
        self.assertTrue(evidence.metadata["drift"])
        self.assertGreater(evidence.metadata["wasserstein_distance"], 3.0)
        self.assertEqual(evidence.direction, "increased")

    def test_stable_target_and_predictions_remain_stable(self):
        target = TargetDriftAnalyzer().analyze(
            self.reference_target, self.reference_target.copy(), "sales"
        )
        prediction = PredictionDriftAnalyzer().analyze(
            LinearModel(), self.reference_features, self.reference_features.copy()
        )

        for evidence in (target, prediction):
            with self.subTest(type=evidence.type):
                self.assertFalse(evidence.metadata["drift"])
                self.assertFalse(evidence.metadata["statistically_significant"])
                self.assertEqual(evidence.value, 0.0)
                self.assertEqual(evidence.severity, "info")

    def test_evidence_serialization_includes_regression_statistics(self):
        serialized = TargetDriftAnalyzer().compare(
            self.reference_target, self.shifted_target, "sales"
        ).to_dict()

        self.assertEqual(serialized["type"], "target_drift")
        self.assertEqual(serialized["sample_size_reference"], 1500)
        for metric in (
            "ks_statistic",
            "psi",
            "wasserstein_distance",
            "reference_mean",
            "production_mean",
            "reference_median",
            "production_median",
            "reference_std",
            "production_std",
        ):
            self.assertIn(metric, serialized["metadata"])

    def test_investigation_output_contains_serialized_distribution_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            reference_path = Path(directory) / "reference.csv"
            production_path = Path(directory) / "production.csv"
            report_path = Path(directory) / "report.html"
            pd.DataFrame({"x": np.arange(20), "target": np.arange(20) * 2}).to_csv(
                reference_path, index=False
            )
            pd.DataFrame({"x": np.arange(20, 40), "target": np.arange(20, 40) * 3}).to_csv(
                production_path, index=False
            )

            runner = ResearchRunner.__new__(ResearchRunner)
            runner.target = "target"
            runner.model = LinearModel()
            runner.experiments = SimpleNamespace(random_state=7, adaptation_fraction=0.5)
            runner.trainer = Mock()
            runner.trainer.evaluate.return_value = {"rmse": 1.0}
            runner.target_drift = TargetDriftAnalyzer()
            runner.prediction_drift = PredictionDriftAnalyzer()
            runner.drift = Mock()
            runner.drift.compare.return_value = []
            runner.drift.diagnose.return_value = {
                "status": "healthy",
                "performance_degradation": [],
                "drifted_features": [],
                "summary": "No input drift.",
            }
            runner.drift_enabled = True
            runner.baseline_metrics = {"rmse": 1.0}
            runner.error_analysis_enabled = True
            runner.error_metrics = ["rmse"]
            runner.baseline_run_id = "baseline"
            runner.baseline_candidates = []
            runner.loop = Mock()
            runner.loop.run.return_value = {
                "_champion": runner.model,
                "experiments": [],
                "decision": {
                    "model": "baseline",
                    "iteration": 0,
                    "used_features": ["x"],
                    "deployment_status": "keep_baseline",
                },
            }
            runner.tracker = Mock()
            runner.profiler = Mock()
            runner.profiler.profile.return_value = {"features": []}
            runner.reporter = Mock()
            runner.reporter.build.return_value = str(report_path)

            result = runner.investigate(reference_path, production_path, report_path)

        self.assertEqual(result["target_drift"]["type"], "target_drift")
        self.assertEqual(result["prediction_drift"]["type"], "prediction_drift")
        self.assertEqual(
            [item["type"] for item in result["evidence"]],
            ["target_drift", "prediction_drift"],
        )
        runner.loop.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
