"""Domain models for statistical evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Literal


EvidenceType = Literal[
    "performance_degradation",
    "feature_drift",
    "target_drift",
    "prediction_drift",
    "residual_drift",
    "feature_error_relationship",
    "segment_degradation",
]
EvidenceSeverity = Literal["info", "low", "medium", "high", "critical"]

EVIDENCE_TYPES = frozenset(
    {
        "performance_degradation",
        "feature_drift",
        "target_drift",
        "prediction_drift",
        "residual_drift",
        "feature_error_relationship",
        "segment_degradation",
    }
)
EVIDENCE_SEVERITIES = frozenset({"info", "low", "medium", "high", "critical"})


def make_evidence_id(
    evidence_type: EvidenceType,
    metric: str,
    *,
    feature: str | None = None,
    segment: str | None = None,
) -> str:
    """Build a stable ID from the semantic identity of a finding."""
    identity = "\x1f".join((evidence_type, metric, feature or "", segment or ""))
    return f"evidence-{sha256(identity.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class Evidence:
    """An immutable, JSON-serializable statistical finding."""

    id: str
    type: EvidenceType
    metric: str
    value: float | None
    feature: str | None = None
    segment: str | None = None
    threshold: float | None = None
    severity: EvidenceSeverity = "info"
    direction: str | None = None
    sample_size_reference: int | None = None
    sample_size_production: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("evidence id must not be blank")
        if self.type not in EVIDENCE_TYPES:
            raise ValueError(f"unsupported evidence type: {self.type}")
        if not self.metric.strip():
            raise ValueError("evidence metric must not be blank")
        if self.severity not in EVIDENCE_SEVERITIES:
            raise ValueError(f"unsupported evidence severity: {self.severity}")
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary containing only JSON-compatible values."""
        return {
            "id": self.id,
            "type": self.type,
            "metric": self.metric,
            "value": self.value,
            "feature": self.feature,
            "segment": self.segment,
            "threshold": self.threshold,
            "severity": self.severity,
            "direction": self.direction,
            "sample_size_reference": self.sample_size_reference,
            "sample_size_production": self.sample_size_production,
            "metadata": _thaw_json(self.metadata),
        }


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("evidence metadata must be a mapping")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("evidence metadata keys must be strings")
        frozen[key] = _freeze_json(item)
    return MappingProxyType(frozen)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"evidence metadata value is not JSON-compatible: {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
