"""Deterministic, non-causal diagnosis from measured investigation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..evidence import Evidence, EvidenceStore


DiagnosisSeverity = Literal["info", "medium", "high"]


@dataclass(frozen=True, slots=True)
class Diagnosis:
    """A structured pattern observed across performance and statistical evidence."""

    id: str
    pattern: str
    severity: DiagnosisSeverity
    summary: str
    evidence_ids: tuple[str, ...] = ()
    degraded_metrics: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "severity": self.severity,
            "summary": self.summary,
            "evidence_ids": list(self.evidence_ids),
            "degraded_metrics": list(self.degraded_metrics),
        }


class DiagnosisEngine:
    """Identify deterministic associations without making causal claims."""

    def diagnose(
        self,
        evidence: EvidenceStore,
        baseline_performance: dict,
        production_performance: dict,
    ) -> list[Diagnosis]:
        if not isinstance(evidence, EvidenceStore):
            raise TypeError("evidence must be an EvidenceStore")
        degraded = self.performance_degradation(
            baseline_performance, production_performance
        )
        degraded_metrics = tuple(item["metric"] for item in degraded)
        feature_drift = _active(evidence.get_by_type("feature_drift"))
        strong_feature_drift = [
            item for item in feature_drift if item.severity in {"medium", "high", "critical"}
        ]
        target_drift = _active(evidence.get_by_type("target_drift"))
        prediction_drift = _active(evidence.get_by_type("prediction_drift"))
        residual_drift = _active(evidence.get_by_type("residual_drift"))
        segment_degradation = _active(evidence.get_by_type("segment_degradation"))
        diagnoses = []

        if degraded and strong_feature_drift:
            diagnoses.append(
                _diagnosis(
                    "performance_with_feature_drift",
                    "performance_degradation_with_strong_feature_drift",
                    "high",
                    "Performance degradation is associated with strong feature drift; "
                    "the evidence suggests investigating the affected inputs.",
                    strong_feature_drift,
                    degraded_metrics,
                )
            )
        if degraded and target_drift:
            diagnoses.append(
                _diagnosis(
                    "performance_with_target_drift",
                    "performance_degradation_with_target_drift",
                    "high",
                    "Performance degradation is observed alongside target drift; this "
                    "is consistent with a changed outcome distribution and warrants investigation.",
                    target_drift,
                    degraded_metrics,
                )
            )
        if degraded and residual_drift:
            diagnoses.append(
                _diagnosis(
                    "performance_with_residual_drift",
                    "performance_degradation_with_residual_drift",
                    "high",
                    "Performance degradation is associated with a changed residual "
                    "distribution and warrants investigation.",
                    residual_drift,
                    degraded_metrics,
                )
            )
        if segment_degradation:
            diagnoses.append(
                _diagnosis(
                    "performance_concentrated_in_segments",
                    "performance_degradation_concentrated_in_segments",
                    "high" if degraded else "medium",
                    "Measured performance degradation is concentrated in segments; the "
                    "evidence suggests prioritizing those segments for investigation.",
                    segment_degradation,
                    degraded_metrics,
                )
            )

        observed_drift = [*feature_drift, *target_drift, *prediction_drift, *residual_drift]
        if not degraded and observed_drift:
            diagnoses.append(
                _diagnosis(
                    "drift_without_performance_degradation",
                    "drift_without_measured_performance_degradation",
                    "medium",
                    "Drift is present without measured performance degradation; the "
                    "evidence suggests continued monitoring.",
                    observed_drift,
                    (),
                )
            )
        if degraded and not feature_drift:
            supporting = [*target_drift, *residual_drift, *segment_degradation]
            diagnoses.append(
                _diagnosis(
                    "degradation_without_covariate_drift",
                    "degradation_without_observed_covariate_drift",
                    "medium",
                    "Performance degradation is present without observed covariate drift; "
                    "unmeasured inputs, labels, targets, and residual patterns warrant investigation.",
                    supporting,
                    degraded_metrics,
                )
            )
        if degraded and not diagnoses:
            diagnoses.append(
                Diagnosis(
                    id="performance_degradation_warrants_investigation",
                    pattern="performance_degradation_warrants_investigation",
                    severity="medium",
                    summary=(
                        "Measured performance degradation warrants investigation; current "
                        "evidence does not show a strong diagnostic association."
                    ),
                    degraded_metrics=degraded_metrics,
                )
            )
        if not diagnoses:
            diagnoses.append(
                Diagnosis(
                    id="no_actionable_pattern",
                    pattern="no_actionable_pattern",
                    severity="info",
                    summary=(
                        "Current evidence does not show measured performance degradation "
                        "or an active drift pattern."
                    ),
                )
            )
        return diagnoses

    def summarize(
        self,
        evidence: EvidenceStore,
        baseline_performance: dict,
        production_performance: dict,
    ) -> dict:
        """Return the legacy diagnosis envelope plus structured diagnosis patterns."""
        degraded = self.performance_degradation(
            baseline_performance, production_performance
        )
        feature_drift = _active(evidence.get_by_type("feature_drift"))
        diagnoses = self.diagnose(
            evidence, baseline_performance, production_performance
        )
        drifted_features = []
        for item in feature_drift:
            serialized = item.to_dict()
            measurement = dict(serialized["metadata"])
            measurement.setdefault("feature", item.feature)
            measurement.setdefault("drift", True)
            measurement.setdefault("severity", item.severity)
            measurement.setdefault("effect_size", item.value)
            drifted_features.append(measurement)
        return {
            "status": "degraded" if degraded else "healthy",
            "performance_degradation": degraded,
            "drifted_features": drifted_features,
            "summary": (
                f"{len(degraded)} metrics degraded; "
                f"{len(feature_drift)} features drifted."
            ),
            "patterns": [item.to_dict() for item in diagnoses],
        }

    @staticmethod
    def performance_degradation(baseline: dict, production: dict) -> list[dict]:
        degraded = []
        for metric, baseline_value in baseline.items():
            production_value = production.get(metric)
            if baseline_value is None or production_value is None:
                continue
            worsened = (
                metric in {"rmse", "mae", "mape"} and production_value > baseline_value
            ) or (metric == "r2" and production_value < baseline_value)
            if worsened:
                degraded.append(
                    {
                        "metric": metric,
                        "baseline": baseline_value,
                        "production": production_value,
                        "change_pct": (
                            (production_value - baseline_value) / abs(baseline_value) * 100
                            if baseline_value
                            else None
                        ),
                    }
                )
        return degraded


def _active(evidence: list[Evidence]) -> list[Evidence]:
    active = []
    for item in evidence:
        measured_drift = item.metadata.get("drift")
        if measured_drift is False:
            continue
        if measured_drift is True or item.severity != "info":
            active.append(item)
    return active


def _diagnosis(
    diagnosis_id,
    pattern,
    severity,
    summary,
    evidence,
    degraded_metrics,
):
    return Diagnosis(
        id=diagnosis_id,
        pattern=pattern,
        severity=severity,
        summary=summary,
        evidence_ids=tuple(sorted(item.id for item in evidence)),
        degraded_metrics=tuple(degraded_metrics),
    )


__all__ = ["Diagnosis", "DiagnosisEngine", "DiagnosisSeverity"]
