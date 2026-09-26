import json
import unittest
from dataclasses import FrozenInstanceError

from ml_sherlock.evidence import EVIDENCE_TYPES, Evidence, make_evidence_id


class EvidenceTests(unittest.TestCase):
    def test_creation(self):
        evidence = Evidence(
            id="drift-age",
            type="feature_drift",
            metric="ks_p_value",
            value=0.002,
            feature="age",
            threshold=0.05,
            severity="high",
            direction="increased",
            sample_size_reference=1000,
            sample_size_production=800,
        )

        self.assertEqual(evidence.type, "feature_drift")
        self.assertEqual(evidence.feature, "age")
        self.assertEqual(evidence.severity, "high")

    def test_immutability(self):
        evidence = Evidence("drift-age", "feature_drift", "ks_statistic", 0.31)

        with self.assertRaises(FrozenInstanceError):
            evidence.value = 0.5

    def test_serialization_is_json_compatible(self):
        evidence = Evidence(
            id="performance-rmse",
            type="performance_degradation",
            metric="rmse_change_pct",
            value=14.2,
            metadata={"windows": ["reference", "production"]},
        )

        serialized = evidence.to_dict()
        self.assertEqual(serialized["metadata"]["windows"], ["reference", "production"])
        self.assertEqual(json.loads(json.dumps(serialized)), serialized)

    def test_metadata_is_copied_and_deeply_immutable(self):
        metadata = {"details": {"affected": ["north"]}}
        evidence = Evidence("segment-north", "segment_degradation", "mae", 4.1, metadata=metadata)
        metadata["details"]["affected"].append("south")

        self.assertEqual(evidence.to_dict()["metadata"], {"details": {"affected": ["north"]}})
        with self.assertRaises(TypeError):
            evidence.metadata["new"] = True
        with self.assertRaises(TypeError):
            evidence.metadata["details"]["affected"] += ("south",)

    def test_optional_fields_default_to_none(self):
        evidence = Evidence("target-mean", "target_drift", "mean_change", None)
        serialized = evidence.to_dict()

        self.assertIsNone(evidence.feature)
        self.assertIsNone(evidence.segment)
        self.assertIsNone(evidence.threshold)
        self.assertEqual(evidence.severity, "info")
        self.assertEqual(serialized["metadata"], {})

    def test_supported_types_and_deterministic_ids(self):
        self.assertEqual(
            EVIDENCE_TYPES,
            {
                "performance_degradation",
                "feature_drift",
                "target_drift",
                "prediction_drift",
                "residual_drift",
                "feature_error_relationship",
                "segment_degradation",
            },
        )
        first = make_evidence_id("feature_drift", "ks_statistic", feature="age")
        second = make_evidence_id("feature_drift", "ks_statistic", feature="age")
        different = make_evidence_id("feature_drift", "ks_statistic", feature="income")
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_rejects_unsupported_type_and_severity(self):
        with self.assertRaisesRegex(ValueError, "unsupported evidence type"):
            Evidence("unknown", "unknown", "score", 1.0)
        with self.assertRaisesRegex(ValueError, "unsupported evidence severity"):
            Evidence("bad-severity", "residual_drift", "mean", 1.0, severity="urgent")


if __name__ == "__main__":
    unittest.main()
