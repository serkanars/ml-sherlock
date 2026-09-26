import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml_sherlock.monitoring.drift import DriftAnalyzer
from ml_sherlock.monitoring.statistics import (
    adjust_p_values,
    categorical_drift_statistics,
    numeric_drift_statistics,
    population_stability_index_categorical,
    population_stability_index_numeric,
)


class DriftEngineV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        size = 1000
        cls.reference = pd.DataFrame(
            {
                "shifted": rng.normal(0, 1, size),
                "stable": rng.normal(10, 2, size),
                "category": rng.choice(["a", "b"], size=size, p=[0.7, 0.3]),
                "missingness": rng.normal(5, 1, size),
            }
        )
        cls.production = pd.DataFrame(
            {
                "shifted": rng.normal(2.5, 1, size),
                "stable": cls.reference["stable"].copy(),
                "category": rng.choice(["a", "b", "c"], size=size, p=[0.5, 0.3, 0.2]),
                "missingness": cls.reference["missingness"].copy(),
            }
        )
        cls.production.loc[:299, "missingness"] = np.nan
        cls.results = {
            item["feature"]: item for item in DriftAnalyzer().compare(cls.reference, cls.production)
        }

    def test_numeric_drift_has_ks_psi_and_wasserstein(self):
        shifted = self.results["shifted"]

        self.assertEqual(shifted["test"], "ks_2samp")
        self.assertEqual(shifted["statistic"], shifted["ks_statistic"])
        self.assertEqual(shifted["p_value"], shifted["ks_p_value"])
        self.assertGreater(shifted["psi"], 0.25)
        self.assertGreater(shifted["wasserstein_distance"], 2.0)
        self.assertGreater(shifted["normalized_wasserstein_distance"], 2.0)
        self.assertTrue(shifted["statistical_significance"])
        self.assertTrue(shifted["drift"])
        self.assertIn(shifted["severity"], {"medium", "high", "critical"})

    def test_stable_numeric_feature_remains_stable(self):
        stable = self.results["stable"]

        self.assertAlmostEqual(stable["ks_statistic"], 0.0)
        self.assertAlmostEqual(stable["psi"], 0.0)
        self.assertAlmostEqual(stable["wasserstein_distance"], 0.0)
        self.assertFalse(stable["statistical_significance"])
        self.assertEqual(stable["effect_magnitude"], "negligible")
        self.assertFalse(stable["drift"])
        self.assertEqual(stable["severity"], "info")

    def test_statistical_significance_alone_does_not_determine_drift(self):
        reference = pd.DataFrame({"x": np.linspace(0, 1, 20_000)})
        production = pd.DataFrame({"x": reference["x"] + 0.02})

        result = DriftAnalyzer().compare(reference, production)[0]

        self.assertTrue(result["statistical_significance"])
        self.assertLess(result["effect_size"], 0.1)
        self.assertFalse(result["drift"])
        self.assertEqual(result["severity"], "info")

    def test_categorical_feature_detects_new_category(self):
        category = self.results["category"]

        self.assertEqual(category["test"], "chi2")
        self.assertEqual(category["statistic"], category["chi2_statistic"])
        self.assertEqual(category["p_value"], category["chi2_p_value"])
        self.assertGreater(category["psi"], 0.1)
        self.assertGreater(category["new_category_rate"], 0.15)
        self.assertEqual(category["reference_cardinality"], 2)
        self.assertEqual(category["production_cardinality"], 3)
        self.assertIn("new_categories", category["decision_reasons"])
        self.assertTrue(category["drift"])

    def test_missingness_increase_is_measured_and_affects_decision(self):
        missingness = self.results["missingness"]

        self.assertEqual(missingness["reference_missing_rate"], 0.0)
        self.assertAlmostEqual(missingness["production_missing_rate"], 0.3)
        self.assertAlmostEqual(missingness["missingness_change"], 0.3)
        self.assertIn("missingness_change", missingness["decision_reasons"])
        self.assertTrue(missingness["drift"])

    def test_descriptive_numeric_statistics_are_available(self):
        result = numeric_drift_statistics(self.reference["shifted"], self.production["shifted"])

        for key in (
            "reference_mean",
            "production_mean",
            "mean_change_pct",
            "reference_median",
            "production_median",
            "reference_std",
            "production_std",
        ):
            self.assertIsNotNone(result[key])

    def test_statistical_helpers_are_reusable(self):
        numeric_psi = population_stability_index_numeric(
            self.reference["shifted"], self.production["shifted"]
        )
        categorical_psi = population_stability_index_categorical(
            self.reference["category"], self.production["category"]
        )
        categorical = categorical_drift_statistics(
            self.reference["category"], self.production["category"]
        )

        self.assertGreater(numeric_psi, 0)
        self.assertGreater(categorical_psi, 0)
        self.assertEqual(categorical["psi"], categorical_psi)

    def test_benjamini_hochberg_adjustment_matches_known_values(self):
        results = [
            {"feature": "a", "p_value": 0.01},
            {"feature": "b", "p_value": 0.04},
            {"feature": "c", "p_value": 0.03},
            {"feature": "d", "p_value": 0.20},
        ]

        adjusted = adjust_p_values(results, alpha=0.05)

        self.assertEqual([item["p_value"] for item in adjusted], [0.01, 0.04, 0.03, 0.20])
        for actual, expected in zip(
            [item["adjusted_p_value"] for item in adjusted],
            [0.04, 0.05333333333333334, 0.05333333333333334, 0.20],
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            [item["statistically_significant"] for item in adjusted],
            [True, False, False, False],
        )

    def test_multiple_testing_can_be_disabled(self):
        adjusted = adjust_p_values(
            [{"feature": "a", "p_value": 0.04}, {"feature": "b", "p_value": 0.20}],
            alpha=0.05,
            method="none",
        )

        self.assertEqual(adjusted[0]["adjusted_p_value"], 0.04)
        self.assertTrue(adjusted[0]["statistically_significant"])
        self.assertEqual(adjusted[0]["multiple_testing"], "none")

    def test_drift_decision_uses_configured_multiple_testing_policy(self):
        reference = pd.DataFrame({f"x{index}": [0.0, 1.0] for index in range(10)})
        production = reference.copy()

        def measurements(series, _production):
            candidate = series.name == "x0"
            return {
                "ks_statistic": 0.15 if candidate else 0.0,
                "ks_p_value": 0.02 if candidate else 0.5,
                "psi": 0.15 if candidate else 0.0,
                "normalized_wasserstein_distance": 0.15 if candidate else 0.0,
                "missingness_change": 0.0,
            }

        with patch(
            "ml_sherlock.monitoring.drift.numeric_drift_statistics",
            side_effect=measurements,
        ):
            corrected = DriftAnalyzer(multiple_testing="benjamini_hochberg").compare(
                reference, production
            )
            uncorrected = DriftAnalyzer(multiple_testing="none").compare(reference, production)

        self.assertEqual(corrected[0]["p_value"], 0.02)
        self.assertAlmostEqual(corrected[0]["adjusted_p_value"], 0.2)
        self.assertFalse(corrected[0]["statistically_significant"])
        self.assertFalse(corrected[0]["drift"])
        self.assertEqual(uncorrected[0]["adjusted_p_value"], 0.02)
        self.assertTrue(uncorrected[0]["statistically_significant"])
        self.assertTrue(uncorrected[0]["drift"])


if __name__ == "__main__":
    unittest.main()
