import unittest

from ml_sherlock.evidence import Evidence, EvidenceRanker, EvidenceStore


def _evidence(
    evidence_id,
    evidence_type="feature_drift",
    *,
    value=0.2,
    severity="medium",
    sample_size=100,
    metadata=None,
):
    return Evidence(
        id=evidence_id,
        type=evidence_type,
        metric="diagnostic_score",
        value=value,
        severity=severity,
        sample_size_reference=sample_size,
        sample_size_production=sample_size,
        metadata=metadata or {},
    )


class EvidenceStoreTests(unittest.TestCase):
    def test_collect_retrieve_filter_and_serialize(self):
        drift = _evidence("drift-1")
        segment = _evidence("segment-1", "segment_degradation")
        store = EvidenceStore([drift])
        store.add(segment)

        self.assertEqual(len(store), 2)
        self.assertIs(store.get_by_id("drift-1"), drift)
        self.assertIs(store.get("drift-1"), drift)
        self.assertIsNone(store.get("missing"))
        self.assertEqual(store.get_by_type("segment_degradation"), [segment])
        self.assertEqual([item["id"] for item in store.to_dict()], ["drift-1", "segment-1"])
        self.assertEqual(store.serialize(), store.to_dict())

    def test_duplicate_ids_are_rejected_without_overwriting(self):
        original = _evidence("same-id", value=0.2)
        store = EvidenceStore([original])

        with self.assertRaisesRegex(ValueError, "duplicate evidence id"):
            store.add(_evidence("same-id", value=0.9))

        self.assertIs(store.get("same-id"), original)

    def test_non_evidence_values_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "Evidence objects"):
            EvidenceStore().add({"id": "not-evidence"})


class EvidenceRankerTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            _evidence(
                "feature-error",
                "feature_error_relationship",
                value=0.82,
                severity="high",
                sample_size=1800,
                metadata={"association_score": 0.82, "adjusted_p_value": 0.001},
            ),
            _evidence(
                "segment",
                "segment_degradation",
                value=145.0,
                severity="high",
                sample_size=900,
                metadata={"degradation_pct": 145.0},
            ),
            _evidence(
                "minor-drift",
                value=0.12,
                severity="low",
                sample_size=80,
                metadata={"effect_size": 0.12, "adjusted_p_value": 0.2},
            ),
            _evidence(
                "critical-residual",
                "residual_drift",
                value=0.9,
                severity="critical",
                sample_size=1500,
                metadata={"comparison": {"adjusted_p_value": 0.0001}},
            ),
        ]

    def test_mixed_evidence_has_deterministic_order(self):
        ranker = EvidenceRanker()
        first = ranker.rank(self.items)
        second = ranker.rank(reversed(self.items))

        self.assertEqual([item.id for item in first], [item.id for item in second])
        self.assertEqual(first[0].id, "critical-residual")
        self.assertEqual(first[-1].id, "minor-drift")

    def test_rank_accepts_store_and_applies_top_n(self):
        ranked = EvidenceRanker().rank(EvidenceStore(self.items), limit=2)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].id, "critical-residual")

    def test_ties_are_broken_by_evidence_id(self):
        same_a = _evidence("a")
        same_b = _evidence("b")

        ranked = EvidenceRanker().rank([same_b, same_a])

        self.assertEqual([item.id for item in ranked], ["a", "b"])

    def test_ranking_is_labelled_as_diagnostic_not_causal(self):
        self.assertIn("without assigning causal certainty", EvidenceRanker.__doc__)


if __name__ == "__main__":
    unittest.main()
