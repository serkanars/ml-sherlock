import json, tempfile
from pathlib import Path
import mlflow
import mlflow.sklearn

class DatasetTracker:
    def __init__(self, tracking_uri, experiment_name):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._pending_dataset = None

    def log_dataset(self, df, name, source, context):
        ds = mlflow.data.from_pandas(df, source=source, name=name)
        self._pending_dataset = ds
        return {"name": name, "source": source, "context": context,
                "digest": getattr(ds, "digest", None),
                "rows": len(df), "columns": len(df.columns)}

    def log_run(self, dataset, metrics, params, model, profile, candidates=None):
        with mlflow.start_run() as run:
            if self._pending_dataset is not None:
                mlflow.log_input(self._pending_dataset, context=dataset["context"])
            mlflow.log_params(params)
            mlflow.log_metrics({k:v for k,v in metrics.items() if v is not None})
            mlflow.log_params({"dataset_rows": dataset["rows"],
                               "dataset_columns": dataset["columns"]})
            if dataset["digest"]:
                mlflow.set_tag("dataset_digest", dataset["digest"])
            mlflow.set_tag("dataset_name", dataset["name"])
            mlflow.set_tag("dataset_source", dataset["source"])
            with tempfile.TemporaryDirectory() as d:
                p = Path(d)/"profile.json"
                p.write_text(json.dumps(profile, indent=2, default=str))
                mlflow.log_artifact(str(p), "data")
            if candidates:
                mlflow.log_dict({"candidates": candidates}, "model-selection/candidates.json")
            # MLflow 3.x defaults to ``skops`` serialization, whose strict
            # type validation rejects this sklearn pipeline on some supported
            # NumPy versions. The model is trained locally in this process, so
            # use MLflow's established cloudpickle format explicitly.
            mlflow.sklearn.log_model(
                model,
                "model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            return run.info.run_id

    def log_research_iteration(self, parent_run_id, result, drift):
        """Persist a scheduler iteration as an independently browseable MLflow run."""
        with mlflow.start_run(run_name=f"research-iteration-{result['iteration']}") as run:
            mlflow.set_tags({
                "ml_sherlock.run_type": "research_iteration",
                "ml_sherlock.parent_run_id": parent_run_id,
                "ml_sherlock.action": result["action"],
                "ml_sherlock.status": result["status"],
                "ml_sherlock.recommended_model": result["recommended_model"],
            })
            mlflow.log_params({
                "iteration": result["iteration"], "action": result["action"],
                "training_seed": result["training_seed"], "selected_model": result["candidate_model"],
                "recommended_model": result["recommended_model"],
                "parent_iteration": result.get("parent_iteration", 0),
                "champion_iteration": result.get("champion_iteration", 0),
                "feature_count": len(result["used_features"]),
            })
            metrics = {f"holdout_baseline_{key}": value for key, value in result["baseline_metrics"].items() if value is not None}
            metrics.update({f"holdout_candidate_{key}": value for key, value in result["candidate_metrics"].items() if value is not None})
            metrics.update({f"holdout_recommended_{key}": value for key, value in result["recommended_metrics"].items() if value is not None})
            metrics["improvement_pct"] = result["improvement_pct"]
            metrics["training_duration_seconds"] = result["training_duration_seconds"]
            mlflow.log_metrics(metrics)
            mlflow.log_dict(result["planner"], "research/plan.json")
            mlflow.log_dict({"candidates": result["candidates"]}, "research/candidates.json")
            mlflow.log_dict({"drift": drift, "used_features": result["used_features"]}, "research/evidence.json")
            mlflow.log_dict(result["feature_importance"], "explainability/feature_importance.json")
            mlflow.sklearn.log_model(
                result["_model"], "model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            return run.info.run_id

    def log_final_report(self, parent_run_id, decision, report, model_path, decision_path):
        with mlflow.start_run(run_name="research-final-report") as run:
            mlflow.set_tags({
                "ml_sherlock.run_type": "research_final_report",
                "ml_sherlock.parent_run_id": parent_run_id,
                "ml_sherlock.final_model": decision["model"],
                "ml_sherlock.final_iteration": decision["iteration"],
                "ml_sherlock.deployment_status": decision["deployment_status"],
            })
            for prefix, values in (("final_baseline", decision.get("final_baseline_metrics", {})),
                                   ("final_candidate", decision.get("final_candidate_metrics", {}))):
                mlflow.log_metrics({f"{prefix}_{key}": value for key, value in values.items() if value is not None})
            artifact_paths = [Path(report), Path(model_path), Path(decision_path)]
            report_data = Path(report).with_suffix(".report-data.json")
            if report_data.is_file():
                artifact_paths.append(report_data)
            for path in artifact_paths:
                mlflow.log_artifact(str(path), "decision")
            return run.info.run_id
