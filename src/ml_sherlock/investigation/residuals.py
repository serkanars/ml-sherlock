"""Deterministic residual analysis for labelled regression data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..evidence import Evidence, make_evidence_id
from ..monitoring.drift import DriftAnalyzer


_QUANTILES = {"p50": 0.50, "p75": 0.75, "p90": 0.90, "p95": 0.95, "p99": 0.99}


class ResidualAnalyzer:
    """Calculate regression errors and compare labelled data windows."""

    def __init__(
        self,
        alpha: float = 0.05,
        multiple_testing: str = "benjamini_hochberg",
        effect_threshold: float = 0.1,
    ):
        self.distribution = DriftAnalyzer(
            alpha=alpha,
            multiple_testing=multiple_testing,
            effect_threshold=effect_threshold,
        )

    def calculate(self, model, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Return row-level predictions and errors while preserving the input index."""
        if target not in data:
            raise ValueError(f"Target '{target}' not found.")

        labels = pd.to_numeric(data[target], errors="raise").astype(float)
        predictions = np.asarray(model.predict(data.drop(columns=[target])))
        if predictions.ndim != 1:
            raise ValueError("Residual analysis requires one-dimensional model predictions.")
        if len(predictions) != len(data):
            raise ValueError("Model prediction count does not match labelled data row count.")

        prediction = pd.Series(predictions, index=data.index, dtype=float)
        residual = labels - prediction
        absolute_error = residual.abs()
        denominator = labels.abs().where(labels != 0)
        relative_error = absolute_error / denominator
        return pd.DataFrame(
            {
                "prediction": prediction,
                "residual": residual,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
            },
            index=data.index,
        )

    def summarize(self, residuals: pd.DataFrame) -> dict:
        """Summarize residual location, spread, and absolute-error tail behavior."""
        required = {"residual", "absolute_error"}
        missing = required - set(residuals.columns)
        if missing:
            raise ValueError(f"Residual data is missing columns: {sorted(missing)}")

        residual = residuals["residual"].dropna()
        absolute_error = residuals["absolute_error"].dropna()
        quantiles = absolute_error.quantile(list(_QUANTILES.values())) if len(absolute_error) else None
        return {
            "sample_size": int(len(residual)),
            "mean_residual": _stat_or_none(residual, "mean"),
            "median_residual": _stat_or_none(residual, "median"),
            "residual_std": float(residual.std(ddof=0)) if len(residual) else None,
            "mean_absolute_error": _stat_or_none(absolute_error, "mean"),
            "error_quantiles": {
                name: float(quantiles.loc[quantile]) if quantiles is not None else None
                for name, quantile in _QUANTILES.items()
            },
        }

    def analyze(
        self,
        model,
        production: pd.DataFrame,
        target: str,
        reference: pd.DataFrame | None = None,
    ) -> dict:
        """Analyze production errors and optionally compare them with reference errors."""
        production_residuals = self.calculate(model, production, target)
        production_summary = self.summarize(production_residuals)
        result = {
            "production": production_summary,
            "reference": None,
            "comparison": None,
            "evidence": None,
        }
        if reference is None:
            return result

        reference_residuals = self.calculate(model, reference, target)
        reference_summary = self.summarize(reference_residuals)
        comparison = self.distribution.compare(
            reference_residuals[["residual"]], production_residuals[["residual"]]
        )[0]
        comparison.update(
            {
                "reference_mean_absolute_error": reference_summary["mean_absolute_error"],
                "production_mean_absolute_error": production_summary["mean_absolute_error"],
                "mean_absolute_error_change_pct": _percentage_change(
                    reference_summary["mean_absolute_error"],
                    production_summary["mean_absolute_error"],
                ),
            }
        )
        result.update(
            {
                "reference": reference_summary,
                "comparison": comparison,
                "evidence": self._evidence(
                    comparison,
                    reference_summary,
                    production_summary,
                ),
            }
        )
        return result

    def compare(self, model, reference: pd.DataFrame, production: pd.DataFrame, target: str) -> dict:
        """Compare residual distributions for two labelled windows."""
        return self.analyze(model, production, target, reference=reference)

    def _evidence(self, comparison, reference_summary, production_summary):
        if not comparison["drift"]:
            return None
        reference_mae = reference_summary["mean_absolute_error"]
        production_mae = production_summary["mean_absolute_error"]
        if reference_mae is None or production_mae is None:
            direction = None
        elif production_mae > reference_mae:
            direction = "increased"
        elif production_mae < reference_mae:
            direction = "decreased"
        else:
            direction = "stable"
        return Evidence(
            id=make_evidence_id("residual_drift", "distribution_effect_size"),
            type="residual_drift",
            metric="distribution_effect_size",
            value=comparison["effect_size"],
            threshold=self.distribution.effect_threshold,
            severity=comparison["severity"],
            direction=direction,
            sample_size_reference=reference_summary["sample_size"],
            sample_size_production=production_summary["sample_size"],
            metadata={
                "comparison": comparison,
                "reference_summary": reference_summary,
                "production_summary": production_summary,
            },
        )


def _stat_or_none(values: pd.Series, statistic: str) -> float | None:
    return float(getattr(values, statistic)()) if len(values) else None


def _percentage_change(reference: float | None, production: float | None) -> float | None:
    if reference is None or production is None or reference == 0:
        return None
    return float((production - reference) / abs(reference) * 100)
