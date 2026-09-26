import unittest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_sherlock.evidence import Evidence
from ml_sherlock.investigation import ResidualAnalyzer


class ResidualAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(91)
        size = 1500
        reference = pd.DataFrame(
            {
                "x": rng.normal(0, 1, size),
                "region": rng.choice(["north", "south"], size=size),
            }
        )
        production = pd.DataFrame(
            {
                "x": rng.normal(0, 1, size),
                "region": rng.choice(["north", "south"], size=size),
            }
        )
        reference_signal = 4 * reference["x"] + (reference["region"] == "north") * 2
        production_signal = 4 * production["x"] + (production["region"] == "north") * 2
        reference["target"] = reference_signal + rng.normal(0, 0.2, size)
        production["target"] = production_signal + rng.normal(0, 3.0, size)

        preprocessor = ColumnTransformer(
            [
                ("numeric", StandardScaler(), ["x"]),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), ["region"]),
            ]
        )
        cls.model = Pipeline(
            [("preprocessor", preprocessor), ("model", LinearRegression())]
        ).fit(reference.drop(columns=["target"]), reference["target"])
        cls.reference = reference
        cls.production = production
        cls.analyzer = ResidualAnalyzer()

    def test_calculate_returns_prediction_and_error_columns(self):
        data = self.reference.head(20).copy()
        data.loc[data.index[0], "target"] = 0.0

        residuals = self.analyzer.calculate(self.model, data, "target")

        self.assertEqual(
            list(residuals.columns),
            ["prediction", "residual", "absolute_error", "relative_error"],
        )
        self.assertTrue(
            np.allclose(residuals["absolute_error"], residuals["residual"].abs())
        )
        self.assertTrue(np.isnan(residuals.iloc[0]["relative_error"]))

    def test_summary_contains_requested_statistics_and_quantiles(self):
        residuals = self.analyzer.calculate(self.model, self.production, "target")
        summary = self.analyzer.summarize(residuals)

        self.assertEqual(summary["sample_size"], len(self.production))
        for name in (
            "mean_residual",
            "median_residual",
            "residual_std",
            "mean_absolute_error",
        ):
            self.assertIsInstance(summary[name], float)
        self.assertEqual(set(summary["error_quantiles"]), {"p50", "p75", "p90", "p95", "p99"})
        self.assertLess(
            summary["error_quantiles"]["p50"], summary["error_quantiles"]["p99"]
        )

    def test_increased_production_noise_generates_residual_drift_evidence(self):
        result = self.analyzer.compare(
            self.model, self.reference, self.production, "target"
        )

        self.assertGreater(
            result["production"]["mean_absolute_error"],
            result["reference"]["mean_absolute_error"] * 5,
        )
        self.assertTrue(result["comparison"]["drift"])
        self.assertGreater(result["comparison"]["psi"], 0.25)
        self.assertGreater(result["comparison"]["wasserstein_distance"], 1.0)
        self.assertIsInstance(result["evidence"], Evidence)
        self.assertEqual(result["evidence"].type, "residual_drift")
        self.assertEqual(result["evidence"].direction, "increased")

    def test_stable_residual_distribution_does_not_generate_evidence(self):
        result = self.analyzer.compare(
            self.model, self.reference, self.reference.copy(), "target"
        )

        self.assertFalse(result["comparison"]["drift"])
        self.assertEqual(result["comparison"]["ks_statistic"], 0.0)
        self.assertEqual(result["comparison"]["wasserstein_distance"], 0.0)
        self.assertIsNone(result["evidence"])

    def test_production_only_analysis_does_not_require_reference_labels(self):
        result = self.analyzer.analyze(self.model, self.production, "target")

        self.assertIsNotNone(result["production"]["mean_absolute_error"])
        self.assertIsNone(result["reference"])
        self.assertIsNone(result["comparison"])
        self.assertIsNone(result["evidence"])


if __name__ == "__main__":
    unittest.main()
