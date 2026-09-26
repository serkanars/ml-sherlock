"""Reusable statistical measurements for dataset drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance


_EPSILON = 1e-6


def missing_value_rate(values: pd.Series) -> float:
    """Return the fraction of missing observations in a series."""
    return float(values.isna().mean()) if len(values) else 0.0


def percentage_change(reference: float, production: float) -> float | None:
    """Return signed percentage change when the reference is non-zero."""
    if not np.isfinite(reference) or not np.isfinite(production) or reference == 0:
        return None
    return float((production - reference) / abs(reference) * 100)


def population_stability_index_numeric(
    reference: pd.Series, production: pd.Series, bins: int = 10
) -> float | None:
    """Calculate PSI using bins learned only from the reference distribution."""
    reference_values = _numeric_values(reference)
    production_values = _numeric_values(production)
    if not len(reference_values) or not len(production_values):
        return None

    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    internal_edges = np.unique(np.quantile(reference_values, quantiles))
    if not len(internal_edges):
        combined = np.unique(np.concatenate((reference_values, production_values)))
        internal_edges = (combined[:-1] + combined[1:]) / 2 if len(combined) > 1 else np.array([])
    edges = np.concatenate(([-np.inf], internal_edges, [np.inf]))
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    production_counts, _ = np.histogram(production_values, bins=edges)
    return _population_stability_index(reference_counts, production_counts)


def population_stability_index_categorical(
    reference: pd.Series, production: pd.Series
) -> float | None:
    """Calculate PSI over the union of observed non-missing categories."""
    reference_values = reference.dropna()
    production_values = production.dropna()
    if reference_values.empty or production_values.empty:
        return None
    categories = pd.concat((reference_values, production_values), ignore_index=True).unique()
    reference_counts = np.array([(reference_values == category).sum() for category in categories])
    production_counts = np.array([(production_values == category).sum() for category in categories])
    return _population_stability_index(reference_counts, production_counts)


def numeric_drift_statistics(reference: pd.Series, production: pd.Series) -> dict:
    """Calculate distribution, location, spread, and missingness measurements."""
    reference_values = _numeric_values(reference)
    production_values = _numeric_values(production)
    reference_missing = missing_value_rate(reference)
    production_missing = missing_value_rate(production)

    if len(reference_values) and len(production_values):
        ks_statistic, ks_p_value = ks_2samp(reference_values, production_values)
        distance = float(wasserstein_distance(reference_values, production_values))
        reference_mean = float(np.mean(reference_values))
        production_mean = float(np.mean(production_values))
        reference_median = float(np.median(reference_values))
        production_median = float(np.median(production_values))
        reference_std = float(np.std(reference_values))
        production_std = float(np.std(production_values))
        normalized_distance = distance / reference_std if reference_std > _EPSILON else None
    else:
        ks_statistic = ks_p_value = distance = None
        reference_mean = _mean_or_none(reference_values)
        production_mean = _mean_or_none(production_values)
        reference_median = _median_or_none(reference_values)
        production_median = _median_or_none(production_values)
        reference_std = _std_or_none(reference_values)
        production_std = _std_or_none(production_values)
        normalized_distance = None

    return {
        "ks_statistic": _float_or_none(ks_statistic),
        "ks_p_value": _float_or_none(ks_p_value),
        "psi": population_stability_index_numeric(reference, production),
        "wasserstein_distance": _float_or_none(distance),
        "normalized_wasserstein_distance": _float_or_none(normalized_distance),
        "reference_mean": reference_mean,
        "production_mean": production_mean,
        "mean_change_pct": (
            percentage_change(reference_mean, production_mean)
            if reference_mean is not None and production_mean is not None
            else None
        ),
        "reference_median": reference_median,
        "production_median": production_median,
        "reference_std": reference_std,
        "production_std": production_std,
        "reference_missing_rate": reference_missing,
        "production_missing_rate": production_missing,
        "missingness_change": production_missing - reference_missing,
    }


def categorical_drift_statistics(reference: pd.Series, production: pd.Series) -> dict:
    """Calculate frequency, category novelty, and missingness measurements."""
    reference_values = reference.dropna()
    production_values = production.dropna()
    reference_missing = missing_value_rate(reference)
    production_missing = missing_value_rate(production)
    categories = pd.concat((reference_values, production_values), ignore_index=True).unique()

    if len(reference_values) and len(production_values) and len(categories) >= 2:
        contingency = np.array(
            [
                [(reference_values == category).sum() for category in categories],
                [(production_values == category).sum() for category in categories],
            ]
        )
        chi2_statistic, chi2_p_value, _, _ = chi2_contingency(contingency)
    elif len(reference_values) and len(production_values):
        chi2_statistic, chi2_p_value = 0.0, 1.0
    else:
        chi2_statistic = chi2_p_value = None

    reference_categories = set(reference_values.unique())
    if len(production_values):
        new_category_rate = float((~production_values.isin(reference_categories)).mean())
    else:
        new_category_rate = None

    return {
        "chi2_statistic": _float_or_none(chi2_statistic),
        "chi2_p_value": _float_or_none(chi2_p_value),
        "psi": population_stability_index_categorical(reference, production),
        "new_category_rate": new_category_rate,
        "reference_cardinality": int(reference_values.nunique()),
        "production_cardinality": int(production_values.nunique()),
        "reference_missing_rate": reference_missing,
        "production_missing_rate": production_missing,
        "missingness_change": production_missing - reference_missing,
    }


def _population_stability_index(reference_counts: np.ndarray, production_counts: np.ndarray) -> float:
    reference_distribution = _smoothed_distribution(reference_counts)
    production_distribution = _smoothed_distribution(production_counts)
    return float(
        np.sum(
            (production_distribution - reference_distribution)
            * np.log(production_distribution / reference_distribution)
        )
    )


def _smoothed_distribution(counts: np.ndarray) -> np.ndarray:
    values = counts.astype(float) + _EPSILON
    return values / values.sum()


def _numeric_values(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)


def _float_or_none(value) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def _std_or_none(values: np.ndarray) -> float | None:
    return float(np.std(values)) if len(values) else None
