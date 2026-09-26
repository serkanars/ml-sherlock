import unittest

import numpy as np
import pandas as pd

from ml_sherlock.evidence import Evidence
from ml_sherlock.investigation import ErrorModelAnalyzer


class ErrorModelAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(812)
        size = 1600
        risk = rng.uniform(0, 1, size)
        features = pd.DataFrame(
            {
                "risk_score": risk,
                "noise": rng.normal(0, 1, size),
                "region": rng.choice(["north", "south", "west"], size=size),
            }
        )
        absolute_error = 0.1 + 7.0 * np.square(risk) + rng.normal(0, 0.08, size)
        signs = rng.choice([-1.0, 1.0], size=size)
        cls.features = features
        cls.primary_predictions = np.zeros(size)
        cls.y_true = absolute_error * signs

    def test_error_model_uses_holdout_and_reports_validation_metrics(self):
        result = ErrorModelAnalyzer(random_state=17).analyze(
            self.features, self.y_true, self.primary_predictions
        )

        split = result["split"]
        self.assertEqual(split["strategy"], "holdout")
        self.assertEqual(split["train_size"] + split["validation_size"], len(self.features))
        self.assertGreater(split["validation_size"], 0)
        self.assertGreater(result["validation_metrics"]["r2"], 0.9)
        self.assertIn("rmse", result["validation_metrics"])
        self.assertIn("mae", result["validation_metrics"])
        self.assertTrue(result["diagnostic_only"])

    def test_error_driving_feature_is_ranked_first(self):
        result = ErrorModelAnalyzer(random_state=17).analyze(
            self.features, self.y_true, self.primary_predictions
        )

        importance = result["feature_importance"]
        self.assertEqual({item["feature"] for item in importance}, set(self.features.columns))
        self.assertEqual(importance[0]["feature"], "risk_score")
        self.assertEqual(importance[0]["rank"], 1)
        self.assertGreater(importance[0]["importance"], 0.9)
        self.assertAlmostEqual(sum(item["importance"] for item in importance), 1.0)

    def test_strong_finding_generates_non_causal_evidence(self):
        result = ErrorModelAnalyzer(random_state=17).analyze(
            self.features, self.y_true, self.primary_predictions
        )
        evidence = result["evidence"][0]

        self.assertIsInstance(evidence, Evidence)
        self.assertEqual(evidence.type, "feature_error_relationship")
        self.assertEqual(evidence.metric, "error_model_feature_importance")
        self.assertEqual(evidence.feature, "risk_score")
        self.assertEqual(evidence.metadata["holdout"]["validation_size"], 320)
        self.assertIn("not a causal explanation", evidence.metadata["interpretation"])

    def test_results_are_deterministic(self):
        analyzer = ErrorModelAnalyzer(random_state=17, n_estimators=40)
        first = analyzer.analyze(self.features, self.y_true, self.primary_predictions)
        second = analyzer.analyze(self.features, self.y_true, self.primary_predictions)

        self.assertEqual(first["split"], second["split"])
        self.assertEqual(first["validation_metrics"], second["validation_metrics"])
        self.assertEqual(first["feature_importance"], second["feature_importance"])

    def test_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            ErrorModelAnalyzer().analyze(
                self.features, self.y_true[:-1], self.primary_predictions
            )


if __name__ == "__main__":
    unittest.main()
