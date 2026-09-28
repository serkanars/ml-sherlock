import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_sherlock.data.tracking import DatasetTracker
from ml_sherlock.investigation.experiments import ExperimentRunner
from ml_sherlock.models.trainer import BaselineTrainer


class ExperimentTrackingTests(unittest.TestCase):
    def test_local_sqlite_parent_is_created(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "artifacts" / "mlflow.db"
            uri = f"sqlite:///{database.as_posix()}"
            with patch("ml_sherlock.data.tracking.mlflow") as mlflow:
                DatasetTracker(uri, "test")
            self.assertTrue(database.parent.is_dir())
            mlflow.set_tracking_uri.assert_called_once_with(uri)

    def test_baseline_run_logs_dataset_input(self):
        tracker = DatasetTracker.__new__(DatasetTracker)
        dataset_input = object()
        tracker._pending_dataset = dataset_input
        dataset = {
            "name": "train", "source": "train.csv", "context": "training",
            "digest": "digest", "rows": 10, "columns": 3,
        }
        with patch("ml_sherlock.data.tracking.mlflow") as mlflow:
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
        with patch("ml_sherlock.data.tracking.mlflow") as mlflow:
            tracker.log_final_report("deleted-parent", decision, "report.html", "model.joblib", "decision.json")
            mlflow.start_run.assert_called_once_with(run_name="research-final-report")
            self.assertEqual(mlflow.set_tags.call_args.args[0]["ml_sherlock.parent_run_id"], "deleted-parent")
            self.assertGreaterEqual(mlflow.log_artifact.call_count, 3)

    def test_final_report_persists_investigation_artifacts_metrics_and_lineage(self):
        tracker = DatasetTracker.__new__(DatasetTracker)
        decision = {
            "model": "lightgbm", "iteration": 2,
            "deployment_status": "review_candidate",
            "training_data": {"reference_path": "reference.csv"},
            "model_path": "model.joblib",
        }
        evidence = [
            {
                "id": "feature-x", "type": "feature_drift", "value": .7,
                "severity": "high", "metadata": {"drift": True},
            },
            {
                "id": "feature-stable", "type": "feature_drift", "value": .01,
                "severity": "info", "metadata": {"drift": False},
            },
            {
                "id": "segment-c", "type": "segment_degradation", "value": 120,
                "severity": "critical", "metadata": {},
            },
            {
                "id": "residual", "type": "residual_drift", "value": .55,
                "severity": "high", "metadata": {"drift": True},
            },
        ]
        diagnosis = {"status": "degraded", "patterns": [{"id": "diagnosis-1"}]}
        hypotheses = [{
            "id": "covariate_shift", "evidence_ids": ["feature-x"],
            "recommended_experiment": "retrain_recent_data",
        }]
        experiments = [{
            "iteration": 2, "hypothesis_id": "covariate_shift",
            "evidence_ids": ["feature-x"],
            "mlflow_run_id": "iteration-run", "action": "retrain_recent_data",
            "status": "validated",
            "dataset_rows_used": {"total_training": 120},
            "used_features": ["x"],
            "candidate_model": "lightgbm",
            "evaluation_metrics": {"candidate": {"rmse": 8.0}},
            "improvement_pct": 20.0,
            "random_seed": 44,
        }]
        dataset = {"name": "reference", "digest": "dataset-digest"}

        with patch("ml_sherlock.data.tracking.mlflow") as mlflow:
            tracker.log_final_report(
                "baseline-run", decision, "report.html", "model.joblib", "decision.json",
                evidence=evidence, diagnosis=diagnosis, hypotheses=hypotheses,
                experiments=experiments, dataset=dataset,
            )

        artifacts = {call.args[1]: call.args[0] for call in mlflow.log_dict.call_args_list}
        self.assertEqual(
            set(artifacts),
            {
                "research/evidence.json", "research/diagnosis.json",
                "research/hypotheses.json", "research/lineage.json",
            },
        )
        self.assertEqual(artifacts["research/evidence.json"]["evidence"], evidence)
        self.assertEqual(artifacts["research/diagnosis.json"], diagnosis)
        lineage = artifacts["research/lineage.json"]
        self.assertEqual(lineage["dataset"]["digest"], "dataset-digest")
        self.assertEqual(lineage["baseline_model"]["mlflow_run_id"], "baseline-run")
        self.assertEqual(lineage["hypotheses"][0]["evidence_ids"], ["feature-x"])
        self.assertEqual(lineage["experiments"][0]["mlflow_run_id"], "iteration-run")
        self.assertEqual(lineage["experiments"][0]["evidence_ids"], ["feature-x"])
        self.assertEqual(lineage["experiments"][0]["dataset_rows_used"]["total_training"], 120)
        self.assertEqual(lineage["experiments"][0]["random_seed"], 44)
        self.assertEqual(lineage["decision"]["iteration"], 2)

        summary = next(
            call.args[0] for call in mlflow.log_metrics.call_args_list
            if "evidence_count" in call.args[0]
        )
        self.assertEqual(summary, {
            "drifted_feature_count": 1.0,
            "severe_drift_count": 2.0,
            "degraded_segment_count": 1.0,
            "evidence_count": 4.0,
            "residual_shift_score": .55,
        })

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
                with patch("ml_sherlock.data.tracking.mlflow") as mlflow:
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
