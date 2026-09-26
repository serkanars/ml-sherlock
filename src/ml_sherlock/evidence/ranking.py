"""Deterministic prioritization of statistical evidence."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from .models import Evidence
from .store import EvidenceStore


_SEVERITY_SCORE = {
    "info": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}


class EvidenceRanker:
    """Rank diagnostic evidence without assigning causal certainty."""

    def rank(
        self,
        evidence: Iterable[Evidence] | EvidenceStore,
        limit: int = 10,
    ) -> list[Evidence]:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        items = list(evidence)
        if any(not isinstance(item, Evidence) for item in items):
            raise TypeError("EvidenceRanker only accepts Evidence objects")
        ranked = sorted(items, key=lambda item: (-self.score(item), item.id))
        return ranked[:limit]

    def score(self, evidence: Evidence) -> float:
        """Return a deterministic diagnostic-priority score in the [0, 1] range."""
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence object")
        severity = _SEVERITY_SCORE[evidence.severity]
        effect = _effect_score(evidence)
        significance = _significance_score(evidence.metadata)
        sample = _sample_score(evidence)
        relationship = _relationship_score(evidence)
        segment = _segment_score(evidence)
        return float(
            0.35 * severity
            + 0.20 * effect
            + 0.15 * significance
            + 0.10 * sample
            + 0.10 * relationship
            + 0.10 * segment
        )


def _effect_score(evidence: Evidence) -> float:
    effect = _find_numeric(evidence.metadata, ("effect_size",))
    if effect is None:
        effect = evidence.value
    if effect is None or not math.isfinite(effect):
        return 0.0
    magnitude = abs(effect)
    if evidence.threshold not in (None, 0) and math.isfinite(evidence.threshold):
        magnitude /= abs(evidence.threshold)
    return magnitude / (1.0 + magnitude)


def _significance_score(metadata: Mapping[str, Any]) -> float:
    adjusted = _find_numeric(metadata, ("adjusted_p_value",))
    if adjusted is None or not math.isfinite(adjusted):
        return 0.0
    return 1.0 - min(1.0, max(0.0, adjusted))


def _sample_score(evidence: Evidence) -> float:
    sizes = [
        size
        for size in (evidence.sample_size_reference, evidence.sample_size_production)
        if size is not None and size > 0
    ]
    if not sizes:
        return 0.0
    effective_size = min(sizes)
    return min(1.0, math.log10(effective_size + 1.0) / 4.0)


def _relationship_score(evidence: Evidence) -> float:
    if evidence.type != "feature_error_relationship":
        return 0.0
    strength = _find_numeric(
        evidence.metadata,
        ("association_score", "feature_importance", "spearman_correlation"),
    )
    if strength is None:
        strength = evidence.value
    return min(1.0, abs(strength)) if strength is not None and math.isfinite(strength) else 0.0


def _segment_score(evidence: Evidence) -> float:
    if evidence.type != "segment_degradation":
        return 0.0
    degradation = _find_numeric(evidence.metadata, ("degradation_pct", "ranking_score"))
    if degradation is None:
        degradation = evidence.value
    if degradation is None or not math.isfinite(degradation):
        return 0.0
    degradation = max(0.0, degradation)
    return degradation / (100.0 + degradation)


def _find_numeric(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    for value in metadata.values():
        if isinstance(value, Mapping):
            found = _find_numeric(value, keys)
            if found is not None:
                return found
    return None


__all__ = ["EvidenceRanker"]
