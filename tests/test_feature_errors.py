import unittest

import numpy as np
import pandas as pd

from ml_sherlock.evidence import Evidence
from ml_sherlock.investigation import FeatureErrorAnalyzer


class FeatureErrorAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(317)
        size = 2000
        risk = rng.uniform(0, 1, size)
        segment = rng.choice(["standard", "fragile"], size=size, p=[0.8, 0.2])
        absolute_error = (
            0.2
            + 4.0 * risk
            + np.where(segment == "fragile", 2.5, 0.0)
            + rng.normal(0, 0.08, size)
        )
        signs = rng.choice([-1.0, 1.0], size=size)
        cls.features = pd.DataFrame(
            {
                "risk_score": risk,
                "noise": rng.normal(0, 1, size),
                "segment": segment,
            }
        )
        cls.predictions = np.zeros(size)
        cls.y_true = absolute_error * signs

    def test_numeric_feature_with_increasing_error_ranks_near_top(self):
        result = FeatureErrorAnalyzer().analyze(
            self.features, self.y_true, self.predictions
        )
        relationships = {item["feature"]: item for item in result["relationships"]}
        risk = relationships["risk_score"]

        self.assertLessEqual(risk["rank"], 2)
        self.assertGreater(risk["spearman_correlation"], 0.75)
        self.assertGreater(risk["association_score"], 0.75)
        self.assertEqual(len(risk["error_by_quantile"]), 5)
        self.assertLess(
            risk["error_by_quantile"][0]["mae"],
            risk["error_by_quantile"][-1]["mae"],
        )

    def test_categorical_relationship_contains_counts_errors_and_lift(self):
        result = FeatureErrorAnalyzer().analyze(
            self.features, self.y_true, self.predictions
        )
        segment = next(
            item for item in result["relationships"] if item["feature"] == "segment"
        )
        groups = {item["group"]: item for item in segment["error_by_category"]}

        self.assertEqual(sum(item["count"] for item in groups.values()), len(self.features))
        self.assertGreater(groups["fragile"]["mae"], groups["standard"]["mae"])
        self.assertGreater(groups["fragile"]["rmse"], groups["standard"]["rmse"])
        self.assertGreater(groups["fragile"]["error_lift"], 1.0)
        self.assertGreater(segment["association_score"], 0.1)

    def test_relationships_are_ranked_deterministically(self):
        analyzer = FeatureErrorAnalyzer(include_mutual_information=True, random_state=19)
        first = analyzer.analyze(self.features, self.y_true, self.predictions)
        second = analyzer.analyze(self.features, self.y_true, self.predictions)

        self.assertEqual(
            [item["feature"] for item in first["relationships"]],
            [item["feature"] for item in second["relationships"]],
        )
        risk = next(item for item in first["relationships"] if item["feature"] == "risk_score")
        self.assertIsInstance(risk["mutual_information"], float)
        self.assertGreater(risk["mutual_information"], 0.5)

    def test_evidence_explains_non_causal_ranking(self):
        result = FeatureErrorAnalyzer().analyze(
            self.features, self.y_true, self.predictions
        )
        evidence = next(item for item in result["evidence"] if item.feature == "risk_score")

        self.assertIsInstance(evidence, Evidence)
        self.assertEqual(evidence.type, "feature_error_relationship")
        self.assertEqual(evidence.metric, "absolute_error_association")
        self.assertEqual(evidence.metadata["rank"], 1)
        self.assertIn("does not establish a causal relationship", evidence.metadata["interpretation"])
        self.assertIn("error_by_quantile", evidence.metadata)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            FeatureErrorAnalyzer().analyze(
                self.features, self.y_true[:-1], self.predictions
            )


if __name__ == "__main__":
    unittest.main()
