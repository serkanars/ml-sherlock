import pandas as pd

from ..evidence import Evidence, make_evidence_id
from .statistics import adjust_p_values, categorical_drift_statistics, numeric_drift_statistics


_EFFECT_THRESHOLD = 0.1
_STRONG_EFFECT_THRESHOLD = 0.25
_MISSINGNESS_THRESHOLD = 0.05
_NEW_CATEGORY_THRESHOLD = 0.01

class DriftAnalyzer:
    def __init__(self, threshold=.05, effect_threshold=_EFFECT_THRESHOLD, *, alpha=None,
                 multiple_testing="benjamini_hochberg"):
        self.alpha = threshold if alpha is None else alpha
        self.threshold = self.alpha
        self.effect_threshold = effect_threshold
        if multiple_testing not in {"benjamini_hochberg", "none"}:
            raise ValueError(f"unsupported multiple-testing method: {multiple_testing}")
        self.multiple_testing = multiple_testing

    def compare(self, reference, production):
        out=[]
        for col in reference.columns:
            if col not in production.columns: continue
            if pd.api.types.is_numeric_dtype(reference[col]):
                measurements = numeric_drift_statistics(reference[col], production[col])
                stat = measurements["ks_statistic"]
                p = measurements["ks_p_value"]
                test = "ks_2samp"
            else:
                measurements = categorical_drift_statistics(reference[col], production[col])
                stat = measurements["chi2_statistic"]
                p = measurements["chi2_p_value"]
                test = "chi2"
            out.append({"feature":col,"test":test,"statistic":stat,
                        "p_value":p, **measurements})

        adjusted = adjust_p_values(out, alpha=self.alpha, method=self.multiple_testing)
        for result in adjusted:
            if result["test"] == "ks_2samp":
                decision = self._numeric_decision(result, result["statistically_significant"])
            else:
                decision = self._categorical_decision(result, result["statistically_significant"])
            result.update(decision)
        return adjusted

    def _numeric_decision(self, measurements, statistically_significant):
        psi = measurements["psi"] or 0.0
        normalized_distance = measurements["normalized_wasserstein_distance"] or 0.0
        ks_statistic = measurements["ks_statistic"] or 0.0
        missingness_change = abs(measurements["missingness_change"])
        distribution_effect = max(psi, normalized_distance, ks_statistic)
        strong_effect = max(psi, normalized_distance) >= _STRONG_EFFECT_THRESHOLD
        material_effect = distribution_effect >= self.effect_threshold
        missingness_drift = missingness_change >= _MISSINGNESS_THRESHOLD
        drift = missingness_drift or strong_effect or (statistically_significant and material_effect)
        reasons = []
        if statistically_significant and material_effect:
            reasons.append("statistically_significant_distribution_shift")
        if strong_effect:
            reasons.append("strong_distribution_effect")
        if missingness_drift:
            reasons.append("missingness_change")
        return _decision_fields(statistically_significant, max(distribution_effect, missingness_change), drift, reasons)

    def _categorical_decision(self, measurements, statistically_significant):
        psi = measurements["psi"] or 0.0
        new_category_rate = measurements["new_category_rate"] or 0.0
        missingness_change = abs(measurements["missingness_change"])
        strong_effect = psi >= _STRONG_EFFECT_THRESHOLD
        material_effect = psi >= self.effect_threshold
        new_category_drift = new_category_rate >= _NEW_CATEGORY_THRESHOLD
        missingness_drift = missingness_change >= _MISSINGNESS_THRESHOLD
        drift = (
            strong_effect
            or new_category_drift
            or missingness_drift
            or (statistically_significant and material_effect)
        )
        reasons = []
        if statistically_significant and material_effect:
            reasons.append("statistically_significant_distribution_shift")
        if strong_effect:
            reasons.append("strong_distribution_effect")
        if new_category_drift:
            reasons.append("new_categories")
        if missingness_drift:
            reasons.append("missingness_change")
        return _decision_fields(
            statistically_significant, max(psi, new_category_rate, missingness_change), drift, reasons
        )

    def diagnose(self, baseline, production, drift):
        degraded=[]
        for m,b in baseline.items():
            p=production.get(m)
            if b is None or p is None: continue
            bad = (m in {"rmse","mae","mape"} and p>b) or (m=="r2" and p<b)
            if bad:
                degraded.append({"metric":m,"baseline":b,"production":p,
                                 "change_pct":((p-b)/abs(b)*100) if b else None})
        drifted=[x for x in drift if x["drift"]]
        return {"status":"degraded" if degraded else "healthy",
                "performance_degradation":degraded,
                "drifted_features":drifted,
                "summary":f"{len(degraded)} metrics degraded; {len(drifted)} features drifted."}


class TargetDriftAnalyzer:
    """Compare reference and production regression-target distributions."""

    def __init__(self, alpha=0.05, multiple_testing="benjamini_hochberg",
                 effect_threshold=_EFFECT_THRESHOLD):
        self.distribution = DriftAnalyzer(
            alpha=alpha,
            multiple_testing=multiple_testing,
            effect_threshold=effect_threshold,
        )

    def analyze(self, reference, production, target_name="target"):
        reference_values = _as_numeric_series(reference, target_name)
        production_values = _as_numeric_series(production, target_name)
        result = self.distribution.compare(
            reference_values.to_frame(), production_values.to_frame()
        )[0]
        return _distribution_evidence(
            "target_drift",
            result,
            reference_values,
            production_values,
            feature=target_name,
            threshold=self.distribution.effect_threshold,
        )

    def compare(self, reference, production, target_name="target"):
        return self.analyze(reference, production, target_name)


class PredictionDriftAnalyzer:
    """Compare a regression model's reference and production predictions."""

    def __init__(self, alpha=0.05, multiple_testing="benjamini_hochberg",
                 effect_threshold=_EFFECT_THRESHOLD):
        self.distribution = DriftAnalyzer(
            alpha=alpha,
            multiple_testing=multiple_testing,
            effect_threshold=effect_threshold,
        )

    def analyze(self, model, reference_features, production_features):
        reference_predictions = _as_numeric_series(
            model.predict(reference_features), "prediction"
        )
        production_predictions = _as_numeric_series(
            model.predict(production_features), "prediction"
        )
        result = self.distribution.compare(
            reference_predictions.to_frame(), production_predictions.to_frame()
        )[0]
        return _distribution_evidence(
            "prediction_drift",
            result,
            reference_predictions,
            production_predictions,
            threshold=self.distribution.effect_threshold,
        )

    def compare(self, model, reference_features, production_features):
        return self.analyze(model, reference_features, production_features)


def _decision_fields(statistically_significant, effect_size, drift, reasons):
    if effect_size < _EFFECT_THRESHOLD:
        magnitude = "negligible"
    elif effect_size < _STRONG_EFFECT_THRESHOLD:
        magnitude = "small"
    elif effect_size < 0.5:
        magnitude = "medium"
    elif effect_size < 1.0:
        magnitude = "large"
    else:
        magnitude = "very_large"

    if not drift:
        severity = "info"
    else:
        severity = {
            "negligible": "low",
            "small": "low",
            "medium": "medium",
            "large": "high",
            "very_large": "critical",
        }[magnitude]
    return {
        "statistical_significance": statistically_significant,
        "effect_size": float(effect_size),
        "effect_magnitude": magnitude,
        "drift": bool(drift),
        "severity": severity,
        "decision_reasons": reasons,
    }


def _as_numeric_series(values, name):
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.ndim != 1:
        raise ValueError(f"{name} drift analysis requires one-dimensional values")
    numeric = pd.to_numeric(series, errors="raise").rename(name)
    return numeric.reset_index(drop=True)


def _distribution_evidence(
    evidence_type, result, reference, production, feature=None, threshold=_EFFECT_THRESHOLD
):
    mean_change = result.get("mean_change_pct")
    if mean_change is None:
        direction = None
    elif mean_change > 0:
        direction = "increased"
    elif mean_change < 0:
        direction = "decreased"
    else:
        direction = "stable"
    return Evidence(
        id=make_evidence_id(
            evidence_type, "distribution_effect_size", feature=feature
        ),
        type=evidence_type,
        metric="distribution_effect_size",
        value=result["effect_size"],
        feature=feature,
        threshold=threshold,
        severity=result["severity"],
        direction=direction,
        sample_size_reference=int(reference.notna().sum()),
        sample_size_production=int(production.notna().sum()),
        metadata=result,
    )
