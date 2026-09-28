"""Reproducible, bounded experiments used to validate research hypotheses."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import SUPPORTED_ACTIONS


REGISTERED_ACTIONS = SUPPORTED_ACTIONS


class ExperimentRunner:
    def __init__(self, trainer, selection_metric="rmse", random_state=42,
                 adaptation_fraction=.5, min_improvement_pct=1.0):
        self.trainer = trainer
        self.selection_metric = selection_metric
        self.random_state = random_state
        self.adaptation_fraction = adaptation_fraction
        self.min_improvement_pct = min_improvement_pct

    def validate_retraining(
        self, reference, production, target, baseline_model,
        action="retrain_recent_data", *, hypothesis_id=None, evidence_ids=None,
    ):
        """Adapt on reference plus production data and evaluate on a fixed holdout."""
        started = perf_counter()
        adaptation, holdout = self._split(production)
        baseline_metrics = self.trainer.evaluate(baseline_model, holdout, target)
        adapted_train = pd.concat([reference, adaptation], ignore_index=True)
        model_names = (
            self.trainer.candidates
            if action == "model_search"
            else [self.trainer.model_name(baseline_model)]
        )
        candidates = self.trainer.fit_and_evaluate(
            adapted_train, holdout, target, model_names
        )
        winner = self.trainer.select_best(candidates)
        features = [column for column in reference.columns if column != target]
        return self._result(
            action=action, started=started, baseline_metrics=baseline_metrics,
            winner=winner,
            candidates=[{"model": item.params["model"], "metrics": item.metrics}
                        for item in candidates],
            used_features=features, reference_rows=len(reference),
            adaptation_rows=len(adaptation), holdout_rows=len(holdout),
            hypothesis_id=hypothesis_id, evidence_ids=evidence_ids,
        )

    def validate_recent_window_retraining(
        self, reference, production, target, baseline_model, *,
        hypothesis_id=None, evidence_ids=None,
    ):
        """Train only on the recent labelled adaptation window."""
        started = perf_counter()
        adaptation, holdout = self._split(production)
        baseline_metrics = self.trainer.evaluate(baseline_model, holdout, target)
        candidates = self.trainer.fit_and_evaluate(
            adaptation, holdout, target,
            [self.trainer.model_name(baseline_model)],
        )
        winner = self.trainer.select_best(candidates)
        features = [column for column in reference.columns if column != target]
        return self._result(
            action="recent_window_retraining", started=started,
            baseline_metrics=baseline_metrics, winner=winner,
            candidates=[{"model": item.params["model"], "metrics": item.metrics}
                        for item in candidates],
            used_features=features, reference_rows=0,
            adaptation_rows=len(adaptation), holdout_rows=len(holdout),
            hypothesis_id=hypothesis_id, evidence_ids=evidence_ids,
        )

    def validate_segment_retraining(
        self, reference, production, target, baseline_model, evidence, *,
        hypothesis_id=None, evidence_ids=None,
    ):
        """Adapt using production rows belonging to the strongest evidenced segment."""
        started = perf_counter()
        adaptation, holdout = self._split(production)
        baseline_metrics = self.trainer.evaluate(baseline_model, holdout, target)
        segment = self._first_segment_evidence(evidence)
        selected = self._segment_rows(adaptation, segment)
        note = None
        if selected.empty:
            selected = adaptation
            note = "No evidenced segment rows were available; used the full adaptation partition."
        adapted_train = pd.concat([reference, selected], ignore_index=True)
        candidates = self.trainer.fit_and_evaluate(
            adapted_train, holdout, target,
            [self.trainer.model_name(baseline_model)],
        )
        winner = self.trainer.select_best(candidates)
        features = [column for column in reference.columns if column != target]
        result = self._result(
            action="segment_retraining", started=started,
            baseline_metrics=baseline_metrics, winner=winner,
            candidates=[{"model": item.params["model"], "metrics": item.metrics}
                        for item in candidates],
            used_features=features, reference_rows=len(reference),
            adaptation_rows=len(selected),
            available_adaptation_rows=len(adaptation), holdout_rows=len(holdout),
            hypothesis_id=hypothesis_id, evidence_ids=evidence_ids,
        )
        result["segment"] = self._evidence_value(segment, "segment") if segment else None
        if note:
            result["note"] = note
        return result

    def validate_drop_drifted_features(
        self, reference, production, target, baseline_model, features, *,
        hypothesis_id=None, evidence_ids=None,
    ):
        """Test whether removing statistically drifted inputs improves holdout error."""
        started = perf_counter()
        adaptation, holdout = self._split(production)
        baseline_metrics = self.trainer.evaluate(baseline_model, holdout, target)
        columns = [column for column in features
                   if column in reference.columns and column != target]
        if not columns or len(columns) == len(reference.columns) - 1:
            result = self.validate_retraining(
                reference, production, target, baseline_model,
                "drop_drifted_features", hypothesis_id=hypothesis_id,
                evidence_ids=evidence_ids,
            )
            result["note"] = (
                "Feature removal would leave no features or remove none; "
                "used retraining fallback."
            )
            return result
        adapted_train = pd.concat([reference, adaptation], ignore_index=True).drop(
            columns=columns
        )
        holdout_without_drift = holdout.drop(columns=columns)
        candidates = self.trainer.fit_and_evaluate(
            adapted_train, holdout_without_drift, target,
            [self.trainer.model_name(baseline_model)],
        )
        winner = self.trainer.select_best(candidates)
        used_features = [column for column in reference.columns
                         if column != target and column not in columns]
        result = self._result(
            action="drop_drifted_features", started=started,
            baseline_metrics=baseline_metrics, winner=winner,
            candidates=[{"model": item.params["model"], "metrics": item.metrics}
                        for item in candidates],
            used_features=used_features, reference_rows=len(reference),
            adaptation_rows=len(adaptation), holdout_rows=len(holdout),
            hypothesis_id=hypothesis_id, evidence_ids=evidence_ids,
        )
        result["dropped_features"] = columns
        return result

    def validate_feature_subset_search(
        self, reference, production, target, baseline_model, evidence,
        fallback_features, *, hypothesis_id=None, evidence_ids=None,
    ):
        """Search a small deterministic set of evidence-derived feature subsets."""
        started = perf_counter()
        adaptation, holdout = self._split(production)
        baseline_metrics = self.trainer.evaluate(baseline_model, holdout, target)
        all_features = [column for column in reference.columns if column != target]
        evidenced = []
        for item in evidence or []:
            feature = self._evidence_value(item, "feature")
            if feature in all_features and feature not in evidenced:
                evidenced.append(feature)
        for feature in fallback_features or []:
            if feature in all_features and feature not in evidenced:
                evidenced.append(feature)
        evidenced = evidenced[:5]

        subsets = [all_features]
        subsets.extend(
            [feature for feature in all_features if feature != excluded]
            for excluded in evidenced
        )
        if evidenced:
            subsets.append([feature for feature in all_features if feature not in evidenced])
        subsets = self._unique_nonempty_subsets(subsets)

        adapted_train = pd.concat([reference, adaptation], ignore_index=True)
        model_name = self.trainer.model_name(baseline_model)
        evaluated = []
        for features in subsets:
            columns = features + [target]
            candidate = self.trainer.fit_and_evaluate(
                adapted_train[columns], holdout[columns], target, [model_name]
            )[0]
            evaluated.append((candidate, features))
        winner = self.trainer.select_best([item[0] for item in evaluated])
        winner_features = next(features for candidate, features in evaluated
                               if candidate is winner)
        candidates = [
            {
                "model": candidate.params["model"],
                "metrics": candidate.metrics,
                "features": list(features),
                "excluded_features": [feature for feature in all_features
                                      if feature not in features],
            }
            for candidate, features in evaluated
        ]
        result = self._result(
            action="feature_subset_search", started=started,
            baseline_metrics=baseline_metrics, winner=winner,
            candidates=candidates, used_features=winner_features,
            reference_rows=len(reference), adaptation_rows=len(adaptation),
            holdout_rows=len(holdout), hypothesis_id=hypothesis_id,
            evidence_ids=evidence_ids,
        )
        result["searched_feature_subsets"] = len(subsets)
        return result

    def run_action(
        self, action, reference, production, target, baseline_model,
        drifted_features, *, hypothesis_id=None, evidence_ids=None, evidence=None,
    ):
        """Execute only explicitly registered actions; action names never become code."""
        handlers = {
            "retrain_recent_data": lambda: self.validate_retraining(
                reference, production, target, baseline_model,
                hypothesis_id=hypothesis_id, evidence_ids=evidence_ids),
            "model_search": lambda: self.validate_retraining(
                reference, production, target, baseline_model, "model_search",
                hypothesis_id=hypothesis_id, evidence_ids=evidence_ids),
            "drop_drifted_features": lambda: self.validate_drop_drifted_features(
                reference, production, target, baseline_model, drifted_features,
                hypothesis_id=hypothesis_id, evidence_ids=evidence_ids),
            "recent_window_retraining": lambda: self.validate_recent_window_retraining(
                reference, production, target, baseline_model,
                hypothesis_id=hypothesis_id, evidence_ids=evidence_ids),
            "segment_retraining": lambda: self.validate_segment_retraining(
                reference, production, target, baseline_model, evidence,
                hypothesis_id=hypothesis_id, evidence_ids=evidence_ids),
            "feature_subset_search": lambda: self.validate_feature_subset_search(
                reference, production, target, baseline_model, evidence,
                drifted_features, hypothesis_id=hypothesis_id,
                evidence_ids=evidence_ids),
        }
        if action not in REGISTERED_ACTIONS or action not in handlers:
            raise ValueError(f"Unsupported experiment action: {action}")
        return handlers[action]()

    def _split(self, production):
        return train_test_split(
            production, train_size=self.adaptation_fraction,
            random_state=self.random_state,
        )

    def _result(
        self, *, action, started, baseline_metrics, winner, candidates,
        used_features, reference_rows, adaptation_rows, holdout_rows,
        hypothesis_id, evidence_ids, available_adaptation_rows=None,
    ):
        improvement = self._improvement(baseline_metrics, winner.metrics)
        accepted = improvement >= self.min_improvement_pct
        recommended_metrics = winner.metrics if accepted else baseline_metrics
        model_name = winner.params["model"]
        rows = {
            "reference_training": int(reference_rows),
            "production_adaptation": int(adaptation_rows),
            "production_holdout": int(holdout_rows),
            "total_training": int(reference_rows + adaptation_rows),
        }
        if available_adaptation_rows is not None:
            rows["production_adaptation_available"] = int(available_adaptation_rows)
        return {
            "name": action,
            "action": action,
            "hypothesis_id": hypothesis_id,
            "evidence_ids": list(evidence_ids or []),
            "status": "validated" if accepted else "rejected",
            "accepted": accepted,
            "holdout_rows": int(holdout_rows),
            "adaptation_rows": int(adaptation_rows),
            "dataset_rows_used": rows,
            "selection_metric": self.selection_metric,
            "baseline_metrics": baseline_metrics,
            "candidates": candidates,
            "recommended_model": model_name if accepted else "baseline",
            "candidate_model": model_name,
            "model": model_name,
            "candidate_metrics": winner.metrics,
            "recommended_metrics": recommended_metrics,
            "evaluation_metrics": {
                "baseline": baseline_metrics,
                "candidate": winner.metrics,
                "recommended": recommended_metrics,
            },
            "improvement_pct": improvement,
            "training_duration_seconds": perf_counter() - started,
            "used_features": list(used_features),
            "feature_importance": self.trainer.feature_importance(winner.model),
            "random_seed": self.trainer.random_state,
            "split_seed": self.random_state,
            "_model": winner.model,
            "success_criterion": (
                f"{self.selection_metric} improves by at least "
                f"{self.min_improvement_pct:.2f}% on unseen production holdout."
            ),
        }

    def _improvement(self, baseline_metrics, candidate_metrics):
        before = baseline_metrics[self.selection_metric]
        after = candidate_metrics[self.selection_metric]
        delta = after - before if self.selection_metric == "r2" else before - after
        return delta / max(abs(before), 1e-12) * 100

    @staticmethod
    def _evidence_value(evidence, key, default=None):
        if evidence is None:
            return default
        if isinstance(evidence, dict):
            return evidence.get(key, default)
        return getattr(evidence, key, default)

    @classmethod
    def _first_segment_evidence(cls, evidence):
        return next((item for item in evidence or []
                     if cls._evidence_value(item, "type") == "segment_degradation"), None)

    @classmethod
    def _segment_rows(cls, frame, evidence):
        if evidence is None:
            return frame.iloc[0:0]
        feature = cls._evidence_value(evidence, "feature")
        metadata = cls._evidence_value(evidence, "metadata", {}) or {}
        definition = metadata.get("definition", {})
        if feature not in frame or not definition:
            return frame.iloc[0:0]
        if "category" in definition:
            values = frame[feature].astype("string").fillna("<missing>")
            return frame.loc[values == str(definition["category"])]
        values = pd.to_numeric(frame[feature], errors="coerce")
        lower = definition.get("lower_bound")
        upper = definition.get("upper_bound")
        closed = definition.get("closed", "right")
        mask = pd.Series(True, index=frame.index)
        if lower is not None and np.isfinite(lower):
            mask &= values >= lower if closed in {"left", "both"} else values > lower
        if upper is not None and np.isfinite(upper):
            mask &= values <= upper if closed in {"right", "both"} else values < upper
        return frame.loc[mask & values.notna()]

    @staticmethod
    def _unique_nonempty_subsets(subsets):
        unique = []
        seen = set()
        for subset in subsets:
            key = tuple(subset)
            if subset and key not in seen:
                unique.append(list(subset))
                seen.add(key)
        return unique


__all__ = ["ExperimentRunner", "REGISTERED_ACTIONS"]
