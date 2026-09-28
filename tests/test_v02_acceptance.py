import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from ml_sherlock import Sherlock


class V02InvestigationAcceptanceTests(unittest.TestCase):
    """Definition of Done for the deterministic ML-Sherlock v0.2 workflow."""

    def test_known_production_failure_is_detected_explained_and_validated(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            workspace = Path(directory)
            reference, production = _known_failure_data()
            reference.to_csv(workspace / "reference.csv", index=False)
            production.to_csv(workspace / "production.csv", index=False)
            tracking_uri = f"sqlite:///{(workspace / 'mlflow.db').as_posix()}"
            client = MlflowClient(tracking_uri=tracking_uri)
            client.create_experiment(
                "v02-acceptance",
                artifact_location=(workspace / "mlflow-artifacts").as_uri(),
            )
            config = workspace / "sherlock.yaml"
            config.write_text(
                _config(tracking_uri), encoding="utf-8"
            )

            result = Sherlock(config=config).investigate()

            investigation = result["investigation"]
            research = investigation["research"]
            evidence = investigation["evidence"]
            evidence_by_id = {item["id"]: item for item in evidence}

            # Detect: global performance, input drift, residuals, and controls.
            degradation = investigation["diagnosis"]["performance_degradation"]
            self.assertIn("rmse", {item["metric"] for item in degradation})
            self.assertGreater(
                investigation["production_metrics"]["rmse"],
                result["fit"]["metrics"]["rmse"] * 3,
            )

            drift_by_feature = {
                item["feature"]: item for item in investigation["drift"]
            }
            self.assertTrue(drift_by_feature["purchase_frequency"]["drift"])
            for control in (
                "stable_age", "stable_tenure", "control_score", "stable_region"
            ):
                with self.subTest(stable_control=control):
                    self.assertFalse(drift_by_feature[control]["drift"])
                    self.assertFalse(
                        drift_by_feature[control]["statistically_significant"]
                    )

            residual = investigation["residual_analysis"]
            self.assertTrue(residual["comparison"]["drift"])
            self.assertGreater(
                residual["production"]["mean_absolute_error"],
                residual["reference"]["mean_absolute_error"] * 3,
            )
            self.assertIsNotNone(residual["evidence"])

            relationships = investigation["feature_error_analysis"]["relationships"]
            purchase_relationship = next(
                item for item in relationships
                if item["feature"] == "purchase_frequency"
            )
            self.assertLessEqual(purchase_relationship["rank"], 3)
            self.assertGreater(purchase_relationship["association_score"], .25)

            segments = investigation["segment_analysis"]["segments"]
            customer_c = next(
                item for item in segments if item["segment"] == "customer_type=C"
            )
            self.assertLessEqual(customer_c["rank"], 3)
            self.assertGreater(customer_c["degradation_pct"], 300)
            self.assertGreater(customer_c["error_lift"], 1.5)

            # Diagnose and hypothesize: every reference points to measured evidence.
            purchase_drift = next(
                item for item in evidence
                if item["type"] == "feature_drift"
                and item["feature"] == "purchase_frequency"
            )
            purchase_error = next(
                item for item in evidence
                if item["type"] == "feature_error_relationship"
                and item["feature"] == "purchase_frequency"
            )
            segment_evidence = next(
                item for item in evidence
                if item["type"] == "segment_degradation"
                and item["segment"] == "customer_type=C"
            )
            residual_evidence = next(
                item for item in evidence if item["type"] == "residual_drift"
            )
            required_evidence = {
                purchase_drift["id"], purchase_error["id"],
                segment_evidence["id"], residual_evidence["id"],
            }
            self.assertTrue(required_evidence <= set(evidence_by_id))

            patterns = investigation["diagnosis"]["patterns"]
            referenced_by_diagnosis = {
                evidence_id
                for pattern in patterns
                for evidence_id in pattern["evidence_ids"]
            }
            self.assertIn(purchase_drift["id"], referenced_by_diagnosis)
            self.assertIn(segment_evidence["id"], referenced_by_diagnosis)
            self.assertIn(residual_evidence["id"], referenced_by_diagnosis)

            hypotheses = research["hypotheses"]
            self.assertNotIn("llm_plan", research)
            self.assertNotIn("llm_plan_error", research)
            self.assertTrue(hypotheses)
            self.assertTrue(any(item["testable"] for item in hypotheses))
            self.assertTrue(
                all(set(item["evidence_ids"]) <= set(evidence_by_id)
                    for item in hypotheses)
            )
            self.assertNotEqual(
                research["recommended_next_experiment"]["name"],
                "scheduled_monitoring",
            )

            # Experiment and validate: an allowed deterministic trial ran on a fixed holdout.
            self.assertTrue(research["experiments"])
            experiment = research["experiments"][0]
            self.assertIn(experiment["status"], {"validated", "rejected"})
            self.assertIn(experiment["hypothesis_id"], {
                item["id"] for item in hypotheses
            })
            self.assertTrue(set(experiment["evidence_ids"]) <= set(evidence_by_id))
            self.assertGreater(experiment["dataset_rows_used"]["production_holdout"], 0)
            self.assertIn(research["decision"]["deployment_status"], {
                "review_candidate", "keep_baseline"
            })

            # Report and tracking: final artifacts reconstruct the investigation locally.
            report = Path(investigation["report"])
            self.assertTrue(report.is_file())
            self.assertGreater(report.stat().st_size, 1000)
            self.assertIn("<html", report.read_text(encoding="utf-8").lower())

            mlflow_experiment = client.get_experiment_by_name("v02-acceptance")
            self.assertIsNotNone(mlflow_experiment)
            final_runs = client.search_runs(
                [mlflow_experiment.experiment_id],
                filter_string=(
                    "tags.`ml_sherlock.run_type` = 'research_final_report'"
                ),
            )
            self.assertEqual(len(final_runs), 1)
            artifact_names = {
                item.path for item in client.list_artifacts(
                    final_runs[0].info.run_id, "research"
                )
            }
            self.assertTrue({
                "research/evidence.json",
                "research/diagnosis.json",
                "research/hypotheses.json",
                "research/lineage.json",
            } <= artifact_names)


def _known_failure_data(rows=1200, seed=2025):
    rng = np.random.default_rng(seed)
    customer_type = np.resize(np.array(["A", "B", "C"]), rows)
    rng.shuffle(customer_type)

    stable_age = np.resize(np.arange(24, 64), rows)
    stable_tenure = np.resize(np.arange(1, 31), rows)
    control_score = np.resize(np.linspace(-1.0, 1.0, 120), rows)
    stable_region = np.resize(np.array(["north", "south", "east", "west"]), rows)
    for values in (stable_age, stable_tenure, control_score, stable_region):
        rng.shuffle(values)

    reference_frequency = np.clip(rng.normal(5.0, 1.0, rows), 1.0, 9.0)
    production_frequency = np.clip(rng.normal(9.0, 1.2, rows), 5.0, 13.0)
    type_effect = pd.Series(customer_type).map({"A": 0.0, "B": 8.0, "C": 16.0}).to_numpy()
    base = (
        100.0
        + 8.0 * reference_frequency
        + type_effect
        + .35 * stable_age
        + .8 * stable_tenure
    )
    reference_target = base + rng.normal(0.0, .8, rows)

    production_target = (
        100.0
        + 8.0 * production_frequency
        + type_effect
        + .35 * stable_age
        + .8 * stable_tenure
        + rng.normal(0.0, .8, rows)
    )
    changed_c = customer_type == "C"
    production_target[changed_c] += 45.0 + 16.0 * production_frequency[changed_c]

    shared = {
        "customer_type": customer_type,
        "stable_age": stable_age,
        "stable_tenure": stable_tenure,
        "control_score": control_score,
        "stable_region": stable_region,
    }
    reference = pd.DataFrame({
        "purchase_frequency": reference_frequency,
        **shared,
        "customer_value": reference_target,
    })
    production = pd.DataFrame({
        "purchase_frequency": production_frequency,
        **shared,
        "customer_value": production_target,
    })
    return reference, production


def _config(tracking_uri):
    return f"""version: 1
data:
  target: customer_value
  train: reference.csv
  production: production.csv
tracking:
  uri: {tracking_uri}
  experiment: v02-acceptance
models:
  candidates: [random_forest]
  selection_metric: rmse
  random_state: 42
investigation:
  drift:
    enabled: true
    alpha: 0.05
    multiple_testing: benjamini_hochberg
  error_analysis:
    enabled: true
    metrics: [rmse, mae, mape, r2]
  segments:
    enabled: true
    columns: [customer_type]
    min_rows: 80
    numeric_bins: 4
    max_segments: 10
experiments:
  max_iterations: 1
  adaptation_fraction: 0.5
  min_improvement_pct: 1.0
  allowed_actions:
    - retrain_recent_data
    - drop_drifted_features
    - model_search
    - segment_retraining
    - recent_window_retraining
    - feature_subset_search
llm:
  enabled: false
report:
  output: artifacts/v02-report.html
"""


if __name__ == "__main__":
    unittest.main()
