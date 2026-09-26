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
                     "drifted_features": drift, "summary": "1 metric degraded; 1 feature drifted."}
        experiment = {
            "iteration": 1, "action": "retrain_recent_data", "status": "validated",
            "parent_iteration": 0, "candidate_model": "random_forest", "recommended_model": "random_forest",
            "baseline_metrics": {"rmse": 12.0}, "candidate_metrics": {"rmse": 8.0},
            "improvement_pct": 33.3, "adaptation_rows": 40, "holdout_rows": 40,
            "used_features": ["x"], "feature_importance": [{"feature": "x", "importance": 1.0}],
            "planner": {"source": "llm", "hypothesis": "Recent data may help.", "rationale": "Measured drift."},
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
            "experiments": [experiment], "decision": decision, "hypotheses": [],
            "initial_candidates": [
                {"model": "random_forest", "metrics": {"rmse": 10.0}, "selected": True},
                {"model": "xgboost", "metrics": {"rmse": 11.0}, "selected": False},
                {"model": "lightgbm", "metrics": {"rmse": 10.5}, "selected": False},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            ReportBuilder().build(output, "target", baseline, production, drift, diagnosis, research)
            content = output.read_text(encoding="utf-8")
            self.assertIn("Random Forest ile kontrollü geçiş değerlendirilebilir.", content)
            self.assertIn("LLM önerisi, nedensellik kanıtı değil", content)
            self.assertIn("Başlangıç model karşılaştırması", content)
            self.assertIn("XGBoost", content)
            self.assertIn("LightGBM", content)
            self.assertIn("data:image/png;base64,", content)
            self.assertNotIn("—", content)
            snapshot = json.loads(output.with_suffix(".report-data.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["research"]["decision"]["iteration"], 1)
