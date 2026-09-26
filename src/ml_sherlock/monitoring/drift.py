import pandas as pd

from .statistics import categorical_drift_statistics, numeric_drift_statistics


_EFFECT_THRESHOLD = 0.1
_STRONG_EFFECT_THRESHOLD = 0.25
_MISSINGNESS_THRESHOLD = 0.05
_NEW_CATEGORY_THRESHOLD = 0.01

class DriftAnalyzer:
    def __init__(self, threshold=.05, effect_threshold=_EFFECT_THRESHOLD):
        self.threshold = threshold
        self.effect_threshold = effect_threshold

    def compare(self, reference, production):
        out=[]
        for col in reference.columns:
            if col not in production.columns: continue
            if pd.api.types.is_numeric_dtype(reference[col]):
                measurements = numeric_drift_statistics(reference[col], production[col])
                stat = measurements["ks_statistic"]
                p = measurements["ks_p_value"]
                test = "ks_2samp"
                decision = self._numeric_decision(measurements)
            else:
                measurements = categorical_drift_statistics(reference[col], production[col])
                stat = measurements["chi2_statistic"]
                p = measurements["chi2_p_value"]
                test = "chi2"
                decision = self._categorical_decision(measurements)
            out.append({"feature":col,"test":test,"statistic":stat,
                        "p_value":p, **measurements, **decision})
        return out

    def _numeric_decision(self, measurements):
        statistically_significant = self._is_significant(measurements["ks_p_value"])
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

    def _categorical_decision(self, measurements):
        statistically_significant = self._is_significant(measurements["chi2_p_value"])
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

    def _is_significant(self, p_value):
        return p_value is not None and p_value < self.threshold

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
