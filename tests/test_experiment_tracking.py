import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.model_selection import train_test_split

from autoresearch.data.tracking import DatasetTracker
from autoresearch.investigation.experiments import ExperimentRunner
from autoresearch.models.trainer import BaselineTrainer


class ExperimentTrackingTests(unittest.TestCase):
    def test_baseline_run_logs_dataset_input(self):
        tracker = DatasetTracker.__new__(DatasetTracker)
        dataset_input = object()
        tracker._pending_dataset = dataset_input
        dataset = {
            "name": "train", "source": "train.csv", "context": "training",
            "digest": "digest", "rows": 10, "columns": 3,
        }
        with patch("autoresearch.data.tracking.mlflow") as mlflow:
            run = mlflow.start_run.return_value.__enter__.return_value
            run.info.run_id = "run-id"
            run_id = tracker.log_run(
                dataset, {"rmse": 1.0}, {"model": "random_forest"}, object(), {"rows": 10}
            )
            mlflow.log_input.assert_called_once_with(dataset_input, context="training")
            self.assertEqual(run_id, "run-id")

    def test_final_report_does_not_reopen_parent_run(self):
        tracker = DatasetTracker.__new__(DatasetTracker)
        decision = {"model": "random_forest", "iteration": 3,
                    "deployment_status": "review_candidate",
                    "final_baseline_metrics": {"rmse": 10},
                    "final_candidate_metrics": {"rmse": 8}}
        with patch("autoresearch.data.tracking.mlflow") as mlflow:
            tracker.log_final_report("deleted-parent", decision, "report.html", "model.joblib", "decision.json")
            mlflow.start_run.assert_called_once_with(run_name="research-final-report")
            self.assertEqual(mlflow.set_tags.call_args.args[0]["ml_sherlock.parent_run_id"], "deleted-parent")
            self.assertGreaterEqual(mlflow.log_artifact.call_count, 3)

    @classmethod
    def setUpClass(cls):
        cls.reference = pd.DataFrame({
            "x": range(40), "z": [i % 3 for i in range(40)],
            "target": [i * 2 for i in range(40)],
        })
        cls.production = pd.DataFrame({
            "x": range(40, 80), "z": [i % 3 for i in range(40, 80)],
            "target": [i * 3 for i in range(40, 80)],
        })
        cls.trainer = BaselineTrainer(random_state=1)
        cls.baseline = cls.trainer.fit_full(cls.reference, "target", "random_forest")

    def test_rejected_candidates_keep_actual_metrics_and_matching_artifact(self):
        runner = ExperimentRunner(self.trainer, random_state=1, min_improvement_pct=1e9)
        _, holdout = train_test_split(self.production, train_size=.5, random_state=1)
        for action in ("retrain_recent_data", "model_search", "drop_drifted_features"):
            with self.subTest(action=action):
                result = runner.run_action(
                    action, self.reference, self.production, "target", self.baseline, ["x"]
                )
                self.assertEqual(result["status"], "rejected")
                self.assertEqual(result["recommended_metrics"], result["baseline_metrics"])
                self.assertNotEqual(result["candidate_metrics"], result["baseline_metrics"])
                evaluation = holdout[result["used_features"] + ["target"]]
                actual = self.trainer.evaluate(result["_model"], evaluation, "target")
                for metric, value in actual.items():
                    self.assertAlmostEqual(result["candidate_metrics"][metric], value)
                expected = set(BaselineTrainer.SUPPORTED_MODELS) if action == "model_search" else {"random_forest"}
                self.assertEqual({c["model"] for c in result["candidates"]}, expected)

                result.update(iteration=1, training_seed=1, planner={"action": action})
                tracker = DatasetTracker.__new__(DatasetTracker)
                with patch("autoresearch.data.tracking.mlflow") as mlflow:
                    tracker.log_research_iteration("parent", result, [])
                    metrics = mlflow.log_metrics.call_args.args[0]
                    self.assertAlmostEqual(metrics["holdout_candidate_rmse"], actual["rmse"])
                    self.assertEqual(metrics["holdout_recommended_rmse"], result["baseline_metrics"]["rmse"])
                    params = mlflow.log_params.call_args.args[0]
                    self.assertEqual(params["selected_model"], result["candidate_model"])
                    self.assertEqual(params["recommended_model"], "baseline")
                    self.assertIs(mlflow.sklearn.log_model.call_args.args[0], result["_model"])

    def test_accepted_candidate_is_also_recommended(self):
        runner = ExperimentRunner(self.trainer, random_state=1)
        result = runner.validate_retraining(self.reference, self.production, "target", self.baseline)
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["candidate_metrics"], result["recommended_metrics"])
        self.assertEqual(result["candidate_model"], result["recommended_model"])
