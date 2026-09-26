"""In-memory collection for evidence produced during one investigation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .models import Evidence, EvidenceType


class EvidenceStore:
    """Collect unique Evidence objects while preserving insertion order."""

    def __init__(self, evidence: Iterable[Evidence] | None = None):
        self._evidence: dict[str, Evidence] = {}
        if evidence is not None:
            self.extend(evidence)

    def add(self, evidence: Evidence) -> Evidence:
        if not isinstance(evidence, Evidence):
            raise TypeError("EvidenceStore only accepts Evidence objects")
        if evidence.id in self._evidence:
            raise ValueError(f"duplicate evidence id: {evidence.id}")
        self._evidence[evidence.id] = evidence
        return evidence

    def extend(self, evidence: Iterable[Evidence]) -> None:
        for item in evidence:
            self.add(item)

    def get_by_id(self, evidence_id: str) -> Evidence | None:
        return self._evidence.get(evidence_id)

    def get(self, evidence_id: str) -> Evidence | None:
        return self.get_by_id(evidence_id)

    def get_by_type(self, evidence_type: EvidenceType) -> list[Evidence]:
        return [item for item in self._evidence.values() if item.type == evidence_type]

    def all(self) -> list[Evidence]:
        return list(self._evidence.values())

    def serialize(self) -> list[dict]:
        return [item.to_dict() for item in self._evidence.values()]

    def to_dict(self) -> list[dict]:
        return self.serialize()

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self._evidence.values())

    def __len__(self) -> int:
        return len(self._evidence)


__all__ = ["EvidenceStore"]
