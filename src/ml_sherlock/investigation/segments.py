"""Deterministic discovery of segments with concentrated model degradation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..evidence import Evidence, make_evidence_id


_SUPPORTED_METRICS = {"rmse", "mae", "mape", "r2"}
_TARGET_COLUMN = "__ml_sherlock_y_true__"
_PREDICTION_COLUMN = "__ml_sherlock_prediction__"


class SegmentAnalyzer:
    """Compare model performance across categorical and numeric production segments."""

    def __init__(
        self,
        min_rows: int = 50,
        numeric_bins: int = 4,
        max_segments: int = 20,
        columns: list[str] | None = None,
    ):
        if min_rows < 2:
            raise ValueError("min_rows must be at least 2")
        if numeric_bins < 2:
            raise ValueError("numeric_bins must be at least 2")
        if max_segments < 1:
            raise ValueError("max_segments must be at least 1")
        self.min_rows = min_rows
        self.numeric_bins = numeric_bins
        self.max_segments = max_segments
        self.columns = list(columns or [])

    @classmethod
    def from_config(cls, config):
        """Construct an analyzer from the typed segment configuration."""
        return cls(
            min_rows=config.min_rows,
            numeric_bins=config.numeric_bins,
            max_segments=config.max_segments,
            columns=list(config.columns),
        )

    def analyze(
        self,
        reference_features: pd.DataFrame,
        production_features: pd.DataFrame,
        reference_y_true,
        reference_predictions,
        production_y_true,
        production_predictions,
        metric: str = "rmse",
    ) -> dict:
        """Return ranked segment degradation measurements and Evidence objects."""
        metric = metric.lower()
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(f"unsupported segment metric: {metric}")
        reference = _evaluation_frame(
            reference_features, reference_y_true, reference_predictions, "reference"
        )
        production = _evaluation_frame(
            production_features, production_y_true, production_predictions, "production"
        )
        columns = self._analysis_columns(reference_features, production_features)
        overall_reference, _ = _metric(
            reference[_TARGET_COLUMN], reference[_PREDICTION_COLUMN], metric
        )
        overall_production, _ = _metric(
            production[_TARGET_COLUMN], production[_PREDICTION_COLUMN], metric
        )
        if overall_reference is None or overall_production is None:
            raise ValueError(f"not enough valid rows to calculate overall {metric}")

        candidates = []
        for feature in columns:
            if (
                pd.api.types.is_numeric_dtype(reference_features[feature])
                and not pd.api.types.is_bool_dtype(reference_features[feature])
            ):
                candidates.extend(
                    self._numeric_segments(feature, reference, production, metric, overall_production)
                )
            else:
                candidates.extend(
                    self._categorical_segments(
                        feature, reference, production, metric, overall_production
                    )
                )

        candidates.sort(
            key=lambda item: (
                -item["ranking_score"],
                -item["degradation_pct"],
                -min(item["reference_row_count"], item["production_row_count"]),
                item["segment"],
            )
        )
        segments = candidates[: self.max_segments]
        for rank, segment in enumerate(segments, start=1):
            segment["rank"] = rank
        evidence = [self._evidence(item, metric) for item in segments if item["degradation_pct"] > 0]
        return {
            "metric": metric,
            "overall_reference_metric": overall_reference,
            "overall_production_metric": overall_production,
            "segments": segments,
            "evidence": evidence,
        }

    def _analysis_columns(self, reference, production):
        if self.columns:
            missing_reference = [column for column in self.columns if column not in reference]
            missing_production = [column for column in self.columns if column not in production]
            if missing_reference or missing_production:
                missing = sorted(set(missing_reference + missing_production))
                raise ValueError(f"segment columns missing from a data window: {missing}")
            return self.columns
        return [column for column in reference.columns if column in production.columns]

    def _categorical_segments(self, feature, reference, production, metric, overall_production):
        reference_values = reference[feature].astype("string").fillna("<missing>")
        production_values = production[feature].astype("string").fillna("<missing>")
        categories = sorted(set(reference_values.unique()) | set(production_values.unique()))
        segments = []
        for category in categories:
            segment = self._evaluate_segment(
                feature=feature,
                segment=f"{feature}={category}",
                feature_type="categorical",
                reference=reference,
                production=production,
                reference_mask=reference_values == category,
                production_mask=production_values == category,
                metric=metric,
                overall_production=overall_production,
                definition={"category": str(category)},
            )
            if segment is not None:
                segments.append(segment)
        return segments

    def _numeric_segments(self, feature, reference, production, metric, overall_production):
        reference_values = pd.to_numeric(reference[feature], errors="coerce")
        production_values = pd.to_numeric(production[feature], errors="coerce")
        valid_reference = reference_values.dropna()
        if valid_reference.nunique() < 2:
            return []
        try:
            _, edges = pd.qcut(
                valid_reference,
                q=min(self.numeric_bins, int(valid_reference.nunique())),
                retbins=True,
                duplicates="drop",
            )
        except ValueError:
            return []
        edges = np.unique(edges.astype(float))
        if len(edges) < 2:
            return []
        edges[0], edges[-1] = -np.inf, np.inf
        reference_bins = pd.cut(reference_values, bins=edges, include_lowest=True)
        production_bins = pd.cut(production_values, bins=edges, include_lowest=True)
        segments = []
        for interval in reference_bins.cat.categories:
            segment = self._evaluate_segment(
                feature=feature,
                segment=f"{feature}={interval}",
                feature_type="numeric",
                reference=reference,
                production=production,
                reference_mask=reference_bins == interval,
                production_mask=production_bins == interval,
                metric=metric,
                overall_production=overall_production,
                definition={
                    "lower_bound": None if np.isneginf(interval.left) else float(interval.left),
                    "upper_bound": None if np.isposinf(interval.right) else float(interval.right),
                    "closed": interval.closed,
                },
            )
            if segment is not None:
                segments.append(segment)
        return segments

    def _evaluate_segment(
        self,
        *,
        feature,
        segment,
        feature_type,
        reference,
        production,
        reference_mask,
        production_mask,
        metric,
        overall_production,
        definition,
    ):
        reference_metric, reference_count = _metric(
            reference.loc[reference_mask, _TARGET_COLUMN],
            reference.loc[reference_mask, _PREDICTION_COLUMN],
            metric,
        )
        production_metric, production_count = _metric(
            production.loc[production_mask, _TARGET_COLUMN],
            production.loc[production_mask, _PREDICTION_COLUMN],
            metric,
        )
        if (
            reference_count < self.min_rows
            or production_count < self.min_rows
            or reference_metric is None
            or production_metric is None
        ):
            return None
        degradation = _degradation_pct(reference_metric, production_metric, metric)
        lift = _error_lift(production_metric, overall_production, metric)
        sample_size = min(reference_count, production_count)
        sample_confidence = min(1.0, math.sqrt(sample_size / (4 * self.min_rows)))
        ranking_score = max(degradation, 0.0) * (0.5 + 0.5 * sample_confidence)
        return {
            "feature": feature,
            "feature_type": feature_type,
            "segment": segment,
            "definition": definition,
            "reference_row_count": reference_count,
            "production_row_count": production_count,
            "reference_metric": reference_metric,
            "production_metric": production_metric,
            f"reference_{metric}": reference_metric,
            f"production_{metric}": production_metric,
            "degradation_pct": degradation,
            "error_lift": lift,
            "sample_confidence": sample_confidence,
            "ranking_score": ranking_score,
        }

    @staticmethod
    def _evidence(segment, metric):
        return Evidence(
            id=make_evidence_id(
                "segment_degradation",
                f"{metric}_degradation_pct",
                feature=segment["feature"],
                segment=segment["segment"],
            ),
            type="segment_degradation",
            metric=f"{metric}_degradation_pct",
            value=segment["degradation_pct"],
            feature=segment["feature"],
            segment=segment["segment"],
            threshold=0.0,
            severity=_severity(segment["degradation_pct"]),
            direction="degraded",
            sample_size_reference=segment["reference_row_count"],
            sample_size_production=segment["production_row_count"],
            metadata={
                **segment,
                "interpretation": (
                    "Observed concentration of model degradation in this segment; "
                    "this does not establish a causal relationship."
                ),
            },
        )


def _evaluation_frame(features, y_true, predictions, name):
    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"{name}_features must be a pandas DataFrame")
    labels = np.asarray(y_true)
    predicted = np.asarray(predictions)
    if labels.ndim != 1 or predicted.ndim != 1:
        raise ValueError(f"{name} labels and predictions must be one-dimensional")
    if len(features) != len(labels) or len(features) != len(predicted):
        raise ValueError(f"{name} features, labels, and predictions must have the same length")
    frame = features.reset_index(drop=True).copy()
    frame[_TARGET_COLUMN] = pd.to_numeric(pd.Series(labels), errors="coerce")
    frame[_PREDICTION_COLUMN] = pd.to_numeric(pd.Series(predicted), errors="coerce")
    return frame


def _metric(y_true, predictions, metric):
    frame = pd.DataFrame({"y_true": y_true, "prediction": predictions}).dropna()
    if metric == "mape":
        frame = frame[frame["y_true"] != 0]
    count = int(len(frame))
    if count == 0 or (metric == "r2" and count < 2):
        return None, count
    residual = frame["y_true"].to_numpy() - frame["prediction"].to_numpy()
    if metric == "rmse":
        value = np.sqrt(np.mean(np.square(residual)))
    elif metric == "mae":
        value = np.mean(np.abs(residual))
    elif metric == "mape":
        value = np.mean(np.abs(residual / frame["y_true"].to_numpy())) * 100
    else:
        denominator = np.sum(np.square(frame["y_true"] - frame["y_true"].mean()))
        value = 1 - np.sum(np.square(residual)) / denominator if denominator > 0 else 0.0
    return float(value), count


def _degradation_pct(reference, production, metric):
    if metric == "r2":
        return float((reference - production) / max(abs(reference), 1e-12) * 100)
    return float((production - reference) / max(abs(reference), 1e-12) * 100)


def _error_lift(segment_production, overall_production, metric):
    if metric == "r2":
        segment_error = max(1.0 - segment_production, 0.0)
        overall_error = max(1.0 - overall_production, 1e-12)
        return float(segment_error / overall_error)
    return float(segment_production / max(abs(overall_production), 1e-12))


def _severity(degradation_pct):
    if degradation_pct < 5:
        return "info"
    if degradation_pct < 10:
        return "low"
    if degradation_pct < 25:
        return "medium"
    if degradation_pct < 50:
        return "high"
    return "critical"
