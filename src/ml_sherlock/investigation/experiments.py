"""Reproducible experiments used to validate research hypotheses."""

from time import perf_counter
import pandas as pd
from sklearn.model_selection import train_test_split


class ExperimentRunner:
    def __init__(self, trainer, selection_metric="rmse", random_state=42,
                 adaptation_fraction=.5, min_improvement_pct=1.0):
        self.trainer = trainer
        self.selection_metric = selection_metric
        self.random_state = random_state
        self.adaptation_fraction = adaptation_fraction
        self.min_improvement_pct = min_improvement_pct

    def validate_retraining(self, reference, production, target, baseline_model, action="retrain_recent_data"):
        """Adapt on one production partition and evaluate on an unseen partition."""
        started = perf_counter()
        adaptation, holdout = train_test_split(
            production, train_size=self.adaptation_fraction, random_state=self.random_state
        )
        baseline_metrics = self.trainer.evaluate(baseline_model, holdout, target)
        adapted_train = pd.concat([reference, adaptation], ignore_index=True)
        model_names = self.trainer.candidates if action == "model_search" else [self.trainer.model_name(baseline_model)]
        candidates = self.trainer.fit_and_evaluate(adapted_train, holdout, target, model_names)
        winner = self.trainer.select_best(candidates)
        before, after = baseline_metrics[self.selection_metric], winner.metrics[self.selection_metric]
        if self.selection_metric == "r2":
            improvement = (after - before) / max(abs(before), 1e-12) * 100
        else:
            improvement = (before - after) / max(abs(before), 1e-12) * 100
        validated = improvement >= self.min_improvement_pct
        return {
            "name": action,
            "action": action,
            "status": "validated" if validated else "rejected",
            "holdout_rows": len(holdout), "adaptation_rows": len(adaptation),
            "selection_metric": self.selection_metric,
            "baseline_metrics": baseline_metrics,
            "candidates": [{"model": item.params["model"], "metrics": item.metrics} for item in candidates],
            "recommended_model": winner.params["model"] if validated else "baseline",
            "candidate_model": winner.params["model"],
            "candidate_metrics": winner.metrics,
            "recommended_metrics": winner.metrics if validated else baseline_metrics,
            "improvement_pct": improvement,
            "training_duration_seconds": perf_counter() - started,
            "used_features": [column for column in reference.columns if column != target],
            "feature_importance": self.trainer.feature_importance(winner.model),
            "_model": winner.model,
            "success_criterion": f"{self.selection_metric} improves by at least {self.min_improvement_pct:.2f}% on unseen production holdout.",
        }

    def validate_drop_drifted_features(self, reference, production, target, baseline_model, features):
        """Test whether removing statistically drifted inputs improves holdout error."""
        started = perf_counter()
        adaptation, holdout = train_test_split(
            production, train_size=self.adaptation_fraction, random_state=self.random_state
        )
        baseline_metrics = self.trainer.evaluate(baseline_model, holdout, target)
        columns = [column for column in features if column in reference.columns and column != target]
        if not columns or len(columns) == len(reference.columns) - 1:
            result = self.validate_retraining(reference, production, target, baseline_model, "drop_drifted_features")
            result["note"] = "Feature removal would leave no features or remove none; used retraining fallback."
            return result
        adapted_train = pd.concat([reference, adaptation], ignore_index=True).drop(columns=columns)
        holdout_without_drift = holdout.drop(columns=columns)
        candidates = self.trainer.fit_and_evaluate(
            adapted_train, holdout_without_drift, target, [self.trainer.model_name(baseline_model)]
        )
        winner = self.trainer.select_best(candidates)
        before, after = baseline_metrics[self.selection_metric], winner.metrics[self.selection_metric]
        improvement = ((after - before) if self.selection_metric == "r2" else (before - after)) / max(abs(before), 1e-12) * 100
        validated = improvement >= self.min_improvement_pct
        return {
            "name": "drop_drifted_features", "action": "drop_drifted_features",
            "status": "validated" if validated else "rejected", "dropped_features": columns,
            "holdout_rows": len(holdout), "adaptation_rows": len(adaptation),
            "selection_metric": self.selection_metric, "baseline_metrics": baseline_metrics,
            "candidates": [{"model": item.params["model"], "metrics": item.metrics} for item in candidates],
            "recommended_model": winner.params["model"] if validated else "baseline",
            "candidate_model": winner.params["model"],
            "candidate_metrics": winner.metrics,
            "recommended_metrics": winner.metrics if validated else baseline_metrics,
            "improvement_pct": improvement,
            "training_duration_seconds": perf_counter() - started,
            "used_features": [column for column in reference.columns if column != target and column not in columns],
            "feature_importance": self.trainer.feature_importance(winner.model),
            "_model": winner.model,
            "success_criterion": f"{self.selection_metric} improves by at least {self.min_improvement_pct:.2f}% on unseen production holdout.",
        }

    def run_action(self, action, reference, production, target, baseline_model, drifted_features):
        if action == "drop_drifted_features":
            return self.validate_drop_drifted_features(reference, production, target, baseline_model, drifted_features)
        if action in {"retrain_recent_data", "model_search"}:
            return self.validate_retraining(reference, production, target, baseline_model, action)
        raise ValueError(f"Unsupported experiment action: {action}")
