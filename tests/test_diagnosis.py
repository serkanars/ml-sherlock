import unittest

from ml_sherlock.evidence import Evidence, EvidenceStore
from ml_sherlock.investigation import Diagnosis, DiagnosisEngine
from ml_sherlock.monitoring.drift import DriftAnalyzer


def _evidence(evidence_id, evidence_type, *, severity="high", drift=True):
    return Evidence(
        id=evidence_id,
        type=evidence_type,
        metric="distribution_effect_size",
        value=0.6,
        feature="x" if evidence_type == "feature_drift" else None,
        segment="group=C" if evidence_type == "segment_degradation" else None,
        severity=severity,
        sample_size_reference=500,
        sample_size_production=500,
        metadata={"drift": drift},
    )


class DiagnosisEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = DiagnosisEngine()
        self.baseline = {"rmse": 1.0, "r2": 0.9}
        self.degraded = {"rmse": 2.0, "r2": 0.7}
        self.healthy = dict(self.baseline)

    def _patterns(self, evidence, production=None):
        diagnoses = self.engine.diagnose(
            EvidenceStore(evidence), self.baseline, production or self.degraded
        )
        self.assertTrue(all(isinstance(item, Diagnosis) for item in diagnoses))
        return {item.pattern: item for item in diagnoses}

    def test_performance_degradation_with_strong_feature_drift(self):
        patterns = self._patterns([_evidence("feature", "feature_drift")])

        diagnosis = patterns["performance_degradation_with_strong_feature_drift"]
        self.assertEqual(diagnosis.evidence_ids, ("feature",))
        self.assertEqual(diagnosis.degraded_metrics, ("rmse", "r2"))

    def test_performance_degradation_with_target_drift(self):
        patterns = self._patterns([_evidence("target", "target_drift")])

        self.assertIn("performance_degradation_with_target_drift", patterns)
        self.assertIn("degradation_without_observed_covariate_drift", patterns)

    def test_performance_degradation_with_residual_drift(self):
        patterns = self._patterns([_evidence("residual", "residual_drift")])

        self.assertIn("performance_degradation_with_residual_drift", patterns)

    def test_performance_degradation_concentrated_in_segments(self):
        patterns = self._patterns([_evidence("segment", "segment_degradation")])

        self.assertIn("performance_degradation_concentrated_in_segments", patterns)

    def test_drift_without_measured_performance_degradation(self):
        patterns = self._patterns(
            [_evidence("feature", "feature_drift", severity="low")],
            production=self.healthy,
        )

        self.assertIn("drift_without_measured_performance_degradation", patterns)

    def test_degradation_without_observed_covariate_drift(self):
        patterns = self._patterns([])

        self.assertEqual(
            list(patterns), ["degradation_without_observed_covariate_drift"]
        )

    def test_degradation_with_only_weak_drift_is_not_reported_as_no_action(self):
        patterns = self._patterns(
            [_evidence("weak-feature", "feature_drift", severity="low")]
        )

        self.assertIn("performance_degradation_warrants_investigation", patterns)
        self.assertNotIn("no_actionable_pattern", patterns)

    def test_segment_degradation_can_reveal_localized_pattern(self):
        patterns = self._patterns(
            [_evidence("segment", "segment_degradation")],
            production=self.healthy,
        )

        diagnosis = patterns["performance_degradation_concentrated_in_segments"]
        self.assertEqual(diagnosis.severity, "medium")

    def test_diagnosis_is_deterministic_and_non_causal(self):
        evidence = [
            _evidence("target", "target_drift"),
            _evidence("feature", "feature_drift"),
        ]
        first = self.engine.diagnose(EvidenceStore(evidence), self.baseline, self.degraded)
        second = self.engine.diagnose(
            EvidenceStore(reversed(evidence)), self.baseline, self.degraded
        )

        self.assertEqual(first, second)
        text = " ".join(item.summary.lower() for item in first)
        self.assertNotIn("caused by", text)
        self.assertNotIn("proves", text)

    def test_summary_preserves_legacy_fields_and_adds_patterns(self):
        store = EvidenceStore([_evidence("feature", "feature_drift")])
        summary = self.engine.summarize(store, self.baseline, self.degraded)

        self.assertEqual(summary["status"], "degraded")
        self.assertEqual(len(summary["performance_degradation"]), 2)
        self.assertEqual(summary["drifted_features"][0]["feature"], "x")
        self.assertEqual(
            summary["patterns"][0]["pattern"],
            "performance_degradation_with_strong_feature_drift",
        )

    def test_drift_analyzer_only_measures_drift(self):
        self.assertFalse(hasattr(DriftAnalyzer(), "diagnose"))


if __name__ == "__main__":
    unittest.main()
