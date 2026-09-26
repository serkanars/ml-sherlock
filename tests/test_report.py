import json
import tempfile
import unittest
from pathlib import Path

from ml_sherlock.reporting.report import ReportBuilder


class ReportBuilderTests(unittest.TestCase):
    def test_report_explains_decision_and_preserves_evidence(self):
        baseline = {"rmse": 10.0, "r2": .5}
        production = {"rmse": 12.0, "r2": .3}
        drift = [{"feature": "x", "test": "ks_2samp", "p_value": .001, "drift": True}]
        diagnosis = {"status": "degraded", "performance_degradation": [{"metric": "rmse"}],
                     "drifted_features": drift, "summary": "1 metric degraded; 1 feature drifted.",
                     "patterns": [{
                         "id": "performance_with_feature_drift",
                         "pattern": "performance_degradation_with_strong_feature_drift",
                         "severity": "high",
                         "summary": "Performance degradation is associated with strong feature drift.",
                         "evidence_ids": ["evidence-feature-x"],
                         "degraded_metrics": ["rmse"],
                     }]}
        experiment = {
            "iteration": 1, "action": "retrain_recent_data", "status": "validated",
            "parent_iteration": 0, "candidate_model": "random_forest", "recommended_model": "random_forest",
            "baseline_metrics": {"rmse": 12.0}, "candidate_metrics": {"rmse": 8.0},
            "improvement_pct": 33.3, "adaptation_rows": 40, "holdout_rows": 40,
            "used_features": ["x"], "feature_importance": [{"feature": "x", "importance": 1.0}],
            "planner": {"source": "llm", "hypothesis": "Recent data may help.", "rationale": "Measured drift."},
            "hypothesis_id": "covariate_shift",
        }
        decision = {
            "model": "random_forest", "iteration": 1, "selection_metric": "rmse",
            "deployment_status": "review_candidate", "final_improvement_pct": 20.0, "final_rows": 20,
            "final_baseline_metrics": {"rmse": 10.0}, "final_candidate_metrics": {"rmse": 8.0},
            "used_features": ["x"], "dropped_features": [], "parameters": {"n_estimators": 300},
            "reference_profile": {"features": [{"name": "x", "mean": 1.0}]},
            "production_profile": {"features": [{"name": "x", "mean": 2.0}]},
            "training_data": {"reference_rows": 100},
        }
        research = {
            "experiments": [experiment], "decision": decision, "hypotheses": [{
                "id": "covariate_shift", "type": "covariate_shift",
                "claim": "Feature drift is associated with performance degradation.",
                "evidence_ids": ["evidence-feature-x"], "confidence": None,
                "testable": True, "recommended_experiment": "retrain_recent_data",
                "metadata": {"exploratory": False, "features": ["x"]},
            }],
            "initial_candidates": [
                {"model": "random_forest", "metrics": {"rmse": 10.0}, "selected": True},
                {"model": "xgboost", "metrics": {"rmse": 11.0}, "selected": False},
                {"model": "lightgbm", "metrics": {"rmse": 10.5}, "selected": False},
            ],
        }
        evidence = [{
            "id": "evidence-feature-x", "type": "feature_drift",
            "metric": "distribution_effect_size", "value": .7,
            "feature": "x", "segment": None, "threshold": .1,
            "severity": "high", "direction": "increased",
            "sample_size_reference": 100, "sample_size_production": 80,
            "metadata": {"drift": True, "effect_size": .7, "adjusted_p_value": .001},
        }, {
            "id": "evidence-target", "type": "target_drift",
            "metric": "distribution_effect_size", "value": .4,
            "feature": "target", "segment": None, "severity": "medium",
            "metadata": {"drift": True},
        }, {
            "id": "evidence-prediction", "type": "prediction_drift",
            "metric": "distribution_effect_size", "value": .3,
            "feature": None, "segment": None, "severity": "medium",
            "metadata": {"drift": True},
        }, {
            "id": "evidence-residual", "type": "residual_drift",
            "metric": "distribution_effect_size", "value": .5,
            "feature": None, "segment": None, "severity": "high",
            "metadata": {"drift": True},
        }, {
            "id": "evidence-error-x", "type": "feature_error_relationship",
            "metric": "absolute_error_association", "value": .8,
            "feature": "x", "segment": None, "severity": "high",
            "metadata": {"association_score": .8},
        }, {
            "id": "evidence-segment-c", "type": "segment_degradation",
            "metric": "rmse_degradation_pct", "value": 120.0,
            "feature": "customer_type", "segment": "customer_type=C", "severity": "critical",
            "metadata": {"production_row_count": 150, "degradation_pct": 120.0},
        }]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            ReportBuilder().build(
                output, "target", baseline, production, drift, diagnosis, research,
                evidence=evidence, ranked_evidence=evidence,
            )
            content = output.read_text(encoding="utf-8")
            self.assertIn("Random Forest ile kontrollü geçiş değerlendirilebilir.", content)
            self.assertIn("LLM önerisi, nedensellik kanıtı değil", content)
            self.assertIn("Başlangıç model karşılaştırması", content)
            self.assertIn("XGBoost", content)
            self.assertIn("LightGBM", content)
            self.assertIn("data:image/png;base64,", content)
            for heading in (
                "1. Performans düşüşü", "2. Öncelikli kanıtlar", "3. Özellik drifti",
                "4. Hedef, tahmin ve artık hata drifti", "5. Özellik-hata ilişkileri",
                "6. En çok bozulan segmentler", "7. Tanı",
                "8. Hipotezler ve kanıt zinciri", "9. Deneyler", "10. Nihai öneri",
            ):
                self.assertIn(heading, content)
            self.assertIn("evidence-feature-x", content)
            self.assertIn("covariate_shift", content)
            self.assertIn("Güncel veriyle eğitim", content)
            self.assertIn("33,30%", content)
            for evidence_id in (
                "evidence-target", "evidence-prediction", "evidence-residual",
                "evidence-error-x", "evidence-segment-c",
            ):
                self.assertIn(evidence_id, content)
            self.assertNotIn("—", content)
            snapshot = json.loads(output.with_suffix(".report-data.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["research"]["decision"]["iteration"], 1)
            self.assertEqual(snapshot["evidence"][0]["id"], "evidence-feature-x")
