import unittest

import numpy as np
import pandas as pd

from ml_sherlock.config import SegmentConfig
from ml_sherlock.evidence import Evidence
from ml_sherlock.investigation import SegmentAnalyzer


class SegmentAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(808)
        size = 1600
        categories = np.array(["A"] * 650 + ["B"] * 600 + ["C"] * 330 + ["tiny"] * 20)
        rng.shuffle(categories)
        reference_risk = rng.uniform(0, 1, size)
        production_risk = rng.uniform(0, 1, size)
        cls.reference_features = pd.DataFrame(
            {"customer_type": categories.copy(), "risk_score": reference_risk}
        )
        cls.production_features = pd.DataFrame(
            {"customer_type": categories.copy(), "risk_score": production_risk}
        )
        reference_error = rng.normal(0, 1, size)
        production_error = rng.normal(0, 1, size)
        production_error[categories == "C"] *= 7
        production_error[production_risk >= np.quantile(production_risk, 0.75)] *= 4
        production_error[categories == "tiny"] *= 20
        cls.reference_y = reference_error
        cls.production_y = production_error
        cls.reference_predictions = np.zeros(size)
        cls.production_predictions = np.zeros(size)

    def analyze(self, **overrides):
        options = {"min_rows": 50, "numeric_bins": 4, "max_segments": 20}
        options.update(overrides)
        analyzer = SegmentAnalyzer(**options)
        return analyzer.analyze(
            self.reference_features,
            self.production_features,
            self.reference_y,
            self.reference_predictions,
            self.production_y,
            self.production_predictions,
            metric="rmse",
        )

    def test_categorical_segment_with_known_degradation_is_identified(self):
        result = self.analyze()
        segment = next(item for item in result["segments"] if item["segment"] == "customer_type=C")

        self.assertGreater(segment["production_rmse"], segment["reference_rmse"] * 4)
        self.assertGreater(segment["degradation_pct"], 300)
        self.assertGreater(segment["error_lift"], 1.0)
        self.assertLessEqual(segment["rank"], 2)

    def test_numeric_quantile_segments_use_reference_boundaries(self):
        result = self.analyze()
        numeric = [item for item in result["segments"] if item["feature"] == "risk_score"]

        self.assertEqual(len(numeric), 4)
        self.assertTrue(all(item["feature_type"] == "numeric" for item in numeric))
        self.assertTrue(any(item["degradation_pct"] > 100 for item in numeric))
        self.assertTrue(all("lower_bound" in item["definition"] for item in numeric))

    def test_tiny_segments_are_filtered_from_both_windows(self):
        result = self.analyze()

        self.assertNotIn("customer_type=tiny", [item["segment"] for item in result["segments"]])
        self.assertTrue(
            all(
                item["reference_row_count"] >= 50 and item["production_row_count"] >= 50
                for item in result["segments"]
            )
        )

    def test_segments_are_ranked_and_limited(self):
        result = self.analyze(max_segments=3)

        self.assertEqual(len(result["segments"]), 3)
        self.assertEqual([item["rank"] for item in result["segments"]], [1, 2, 3])
        self.assertEqual(
            [item["ranking_score"] for item in result["segments"]],
            sorted(
                [item["ranking_score"] for item in result["segments"]], reverse=True
            ),
        )

    def test_segment_degradation_evidence_is_generated(self):
        result = self.analyze()
        evidence = next(item for item in result["evidence"] if item.segment == "customer_type=C")

        self.assertIsInstance(evidence, Evidence)
        self.assertEqual(evidence.type, "segment_degradation")
        self.assertEqual(evidence.feature, "customer_type")
        self.assertEqual(evidence.metric, "rmse_degradation_pct")
        self.assertIn("does not establish a causal relationship", evidence.metadata["interpretation"])

    def test_typed_configuration_constructs_analyzer(self):
        config = SegmentConfig(
            enabled=True,
            columns=["customer_type"],
            min_rows=100,
            numeric_bins=5,
            max_segments=7,
        )

        analyzer = SegmentAnalyzer.from_config(config)

        self.assertEqual(analyzer.columns, ["customer_type"])
        self.assertEqual(analyzer.min_rows, 100)
        self.assertEqual(analyzer.numeric_bins, 5)
        self.assertEqual(analyzer.max_segments, 7)


if __name__ == "__main__":
    unittest.main()
