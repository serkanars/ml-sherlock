"""Typed, evidence-traceable hypotheses for deterministic investigation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..evidence import Evidence
from .diagnosis import Diagnosis


@dataclass
class Hypothesis:
    id: str
    type: str
    claim: str
    evidence_ids: list[str]
    confidence: float | None
    testable: bool
    recommended_experiment: str | None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "claim": self.claim,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "testable": self.testable,
            "recommended_experiment": self.recommended_experiment,
            "metadata": dict(self.metadata),
        }


class HypothesisEngine:
    """Create ordered, testable hypotheses from diagnoses and ranked evidence."""

    _PATTERNS = {
        "performance_degradation_with_strong_feature_drift": {
            "type": "covariate_shift",
            "claim": (
                "Measured feature drift is associated with performance degradation; "
                "testing adaptation to representative recent data is warranted."
            ),
            "experiment": "retrain_recent_data",
        },
        "performance_degradation_concentrated_in_segments": {
            "type": "segment_specific_degradation",
            "claim": (
                "Measured degradation is concentrated in identified segments; a "
                "segment-aware validation experiment is warranted."
            ),
            "experiment": "retrain_recent_data",
        },
        "performance_degradation_with_target_drift": {
            "type": "target_relationship_shift",
            "claim": (
                "Target drift is observed alongside performance degradation; testing "
                "with representative recent labels is warranted."
            ),
            "experiment": "retrain_recent_data",
        },
        "performance_degradation_with_residual_drift": {
            "type": "model_family_robustness",
            "claim": (
                "Residual drift is associated with degraded performance; comparing "
                "model families on a fixed holdout is warranted."
            ),
            "experiment": "model_search",
        },
        "drift_without_measured_performance_degradation": {
            "type": "unstable_feature",
            "claim": (
                "Distribution instability is present without measured degradation; "
                "continued labelled-window monitoring can test whether it persists."
            ),
            "experiment": "drift_watch",
        },
    }
    _EXPLORATORY_PATTERNS = {
        "degradation_without_observed_covariate_drift",
        "performance_degradation_warrants_investigation",
    }

    def generate(
        self, diagnosis: Diagnosis, ranked_evidence: list[Evidence]
    ) -> list[Hypothesis]:
        if not isinstance(diagnosis, Diagnosis):
            raise TypeError("diagnosis must be a Diagnosis object")
        if any(not isinstance(item, Evidence) for item in ranked_evidence):
            raise TypeError("ranked_evidence must contain Evidence objects")

        evidence_by_id = {item.id: item for item in ranked_evidence}
        evidence_ids = [
            item.id for item in ranked_evidence if item.id in diagnosis.evidence_ids
        ]
        specification = self._PATTERNS.get(diagnosis.pattern)
        exploratory = diagnosis.pattern in self._EXPLORATORY_PATTERNS
        if specification is None and not exploratory:
            return []
        if not evidence_ids and not exploratory:
            return []

        if exploratory:
            specification = {
                "type": "model_family_robustness",
                "claim": (
                    "Observed degradation lacks a strong measured covariate association; "
                    "an exploratory fixed-holdout model-family comparison is warranted."
                ),
                "experiment": "model_search",
            }
        referenced = [evidence_by_id[item_id] for item_id in evidence_ids]
        metadata = {
            "diagnosis_id": diagnosis.id,
            "diagnosis_pattern": diagnosis.pattern,
            "degraded_metrics": list(diagnosis.degraded_metrics),
            "exploratory": exploratory,
            "features": _unique(item.feature for item in referenced if item.feature),
            "segments": _unique(item.segment for item in referenced if item.segment),
        }
        return [
            Hypothesis(
                id=specification["type"],
                type=specification["type"],
                claim=specification["claim"],
                evidence_ids=evidence_ids,
                confidence=None,
                testable=True,
                recommended_experiment=specification["experiment"],
                metadata=metadata,
            )
        ]

    def generate_many(
        self, diagnoses: list[Diagnosis], ranked_evidence: list[Evidence]
    ) -> list[Hypothesis]:
        evidence_rank = {item.id: rank for rank, item in enumerate(ranked_evidence)}
        candidates = []
        for diagnosis_order, diagnosis in enumerate(diagnoses):
            for hypothesis in self.generate(diagnosis, ranked_evidence):
                first_evidence_rank = min(
                    (evidence_rank[item_id] for item_id in hypothesis.evidence_ids),
                    default=len(ranked_evidence),
                )
                candidates.append((first_evidence_rank, diagnosis_order, hypothesis))
        candidates.sort(key=lambda item: (item[0], item[1], item[2].id))

        hypotheses = []
        seen = set()
        for _, _, hypothesis in candidates:
            if hypothesis.id not in seen:
                hypotheses.append(hypothesis)
                seen.add(hypothesis.id)
        return hypotheses


def _unique(values):
    return list(dict.fromkeys(values))


__all__ = ["Hypothesis", "HypothesisEngine"]
