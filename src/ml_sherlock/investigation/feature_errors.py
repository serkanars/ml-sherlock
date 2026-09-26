"""Feature-level associations with observed regression error."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from ..evidence import Evidence, make_evidence_id
from ..monitoring.statistics import adjust_p_values


class FeatureErrorAnalyzer:
    """Rank non-causal associations between production features and absolute error."""

    def __init__(
        self,
        include_mutual_information: bool = False,
        random_state: int = 42,
        quantile_bins: int = 5,
        min_association: float = 0.1,
        min_category_count: int = 5,
        alpha: float = 0.05,
        multiple_testing: str = "benjamini_hochberg",
    ):
        if quantile_bins < 2:
            raise ValueError("quantile_bins must be at least 2")
        if min_association < 0:
            raise ValueError("min_association must be non-negative")
        if min_category_count < 1:
            raise ValueError("min_category_count must be at least 1")
        self.include_mutual_information = include_mutual_information
        self.random_state = random_state
        self.quantile_bins = quantile_bins
        self.min_association = min_association
        self.min_category_count = min_category_count
        self.alpha = alpha
        self.multiple_testing = multiple_testing

    def analyze(self, features: pd.DataFrame, y_true, predictions) -> dict:
        """Return ranked feature/error relationships and supporting Evidence objects."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        absolute_error = _absolute_error(y_true, predictions, len(features))
        overall_mae = float(absolute_error.dropna().mean())
        if not np.isfinite(overall_mae):
            raise ValueError("at least one valid target and prediction pair is required")

        relationships = []
        for feature in features.columns:
            values = features[feature].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
                relationship = self._numeric_relationship(feature, values, absolute_error, overall_mae)
            else:
                relationship = self._categorical_relationship(
                    feature, values, absolute_error, overall_mae
                )
            relationships.append(relationship)

        relationships = adjust_p_values(
            relationships, alpha=self.alpha, method=self.multiple_testing
        )
        relationships.sort(key=lambda item: (-item["association_score"], item["feature"]))

        evidence = []
        for rank, relationship in enumerate(relationships, start=1):
            relationship["rank"] = rank
            if relationship["association_score"] >= self.min_association:
                evidence.append(self._evidence(relationship, overall_mae))
        return {
            "overall_mean_absolute_error": overall_mae,
            "relationships": relationships,
            "evidence": evidence,
        }

    def _numeric_relationship(self, feature, values, absolute_error, overall_mae):
        frame = pd.DataFrame({"feature": values, "absolute_error": absolute_error}).dropna()
        if len(frame) >= 3 and frame["feature"].nunique() > 1 and frame["absolute_error"].nunique() > 1:
            correlation, p_value = spearmanr(frame["feature"], frame["absolute_error"])
            correlation = float(correlation) if np.isfinite(correlation) else None
            p_value = float(p_value) if np.isfinite(p_value) else None
        else:
            correlation = p_value = None

        mutual_information = None
        if self.include_mutual_information and len(frame) >= 4 and frame["feature"].nunique() > 1:
            mutual_information = float(
                mutual_info_regression(
                    frame[["feature"]],
                    frame["absolute_error"],
                    random_state=self.random_state,
                )[0]
            )

        summaries = _quantile_error_summaries(
            frame, self.quantile_bins, overall_mae
        )
        lift_effect = _maximum_lift_deviation(summaries, self.min_category_count)
        association_score = max(abs(correlation or 0.0), lift_effect)
        return {
            "feature": feature,
            "feature_type": "numeric",
            "sample_size": int(len(frame)),
            "association_score": float(association_score),
            "spearman_correlation": correlation,
            "p_value": p_value,
            "mutual_information": mutual_information,
            "error_by_quantile": summaries,
            "error_by_category": None,
        }

    def _categorical_relationship(self, feature, values, absolute_error, overall_mae):
        categories = values.astype("string").fillna("<missing>")
        frame = pd.DataFrame({"category": categories, "absolute_error": absolute_error}).dropna(
            subset=["absolute_error"]
        )
        summaries = _grouped_error_summaries(frame, "category", overall_mae)
        association_score = _maximum_lift_deviation(summaries, self.min_category_count)
        return {
            "feature": feature,
            "feature_type": "categorical",
            "sample_size": int(len(frame)),
            "association_score": float(association_score),
            "spearman_correlation": None,
            "p_value": None,
            "mutual_information": None,
            "error_by_quantile": None,
            "error_by_category": summaries,
        }

    def _evidence(self, relationship, overall_mae):
        correlation = relationship["spearman_correlation"]
        if correlation is None:
            direction = None
        elif correlation > 0:
            direction = "positive"
        elif correlation < 0:
            direction = "negative"
        else:
            direction = "stable"
        metadata = dict(relationship)
        metadata.update(
            {
                "overall_mean_absolute_error": overall_mae,
                "interpretation": (
                    "Observed association between this feature and absolute model error; "
                    "this does not establish a causal relationship."
                ),
            }
        )
        return Evidence(
            id=make_evidence_id(
                "feature_error_relationship",
                "absolute_error_association",
                feature=relationship["feature"],
            ),
            type="feature_error_relationship",
            metric="absolute_error_association",
            value=relationship["association_score"],
            feature=relationship["feature"],
            threshold=self.min_association,
            severity=_severity(relationship["association_score"]),
            direction=direction,
            sample_size_production=relationship["sample_size"],
            metadata=metadata,
        )


def _absolute_error(y_true, predictions, expected_size):
    labels = np.asarray(y_true)
    predicted = np.asarray(predictions)
    if labels.ndim != 1 or predicted.ndim != 1:
        raise ValueError("y_true and predictions must be one-dimensional")
    if len(labels) != expected_size or len(predicted) != expected_size:
        raise ValueError("features, y_true, and predictions must have the same length")
    labels = pd.to_numeric(pd.Series(labels), errors="coerce")
    predicted = pd.to_numeric(pd.Series(predicted), errors="coerce")
    return (labels - predicted).abs()


def _quantile_error_summaries(frame, bins, overall_mae):
    if frame.empty or frame["feature"].nunique() < 2:
        return []
    bin_count = min(bins, int(frame["feature"].nunique()))
    try:
        quantiles = pd.qcut(frame["feature"], q=bin_count, duplicates="drop")
    except ValueError:
        return []
    grouped = frame.assign(quantile=quantiles)
    summaries = _grouped_error_summaries(grouped, "quantile", overall_mae)
    for summary, (_, group) in zip(summaries, grouped.groupby("quantile", observed=True, sort=True)):
        summary["feature_min"] = float(group["feature"].min())
        summary["feature_max"] = float(group["feature"].max())
    return summaries


def _grouped_error_summaries(frame, group_column, overall_mae):
    summaries = []
    for group_name, group in frame.groupby(group_column, observed=True, sort=True):
        errors = group["absolute_error"]
        mae = float(errors.mean())
        summaries.append(
            {
                "group": str(group_name),
                "count": int(len(group)),
                "mae": mae,
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "error_lift": float(mae / overall_mae) if overall_mae > 0 else None,
            }
        )
    return summaries


def _maximum_lift_deviation(summaries, min_count):
    return max(
        (
            abs(summary["error_lift"] - 1.0)
            for summary in summaries
            if summary["count"] >= min_count and summary["error_lift"] is not None
        ),
        default=0.0,
    )


def _severity(score):
    if score < 0.1:
        return "info"
    if score < 0.25:
        return "low"
    if score < 0.5:
        return "medium"
    if score < 1.0:
        return "high"
    return "critical"
