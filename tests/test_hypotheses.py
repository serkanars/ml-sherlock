import unittest

from ml_sherlock.evidence import Evidence
from ml_sherlock.investigation import Diagnosis, Hypothesis, HypothesisEngine
from ml_sherlock.investigation.engine import ResearchEngine
from ml_sherlock.investigation.loop import ResearchLoop


def _evidence(evidence_id, evidence_type, *, feature=None, segment=None):
    return Evidence(
        id=evidence_id,
        type=evidence_type,
        metric="effect",
        value=0.8,
        feature=feature,
        segment=segment,
        severity="high",
    )


class HypothesisEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = HypothesisEngine()

    def test_covariate_hypothesis_is_typed_and_traceable(self):
        evidence = _evidence("feature-1", "feature_drift", feature="income")
        diagnosis = Diagnosis(
            id="diagnosis-1",
            pattern="performance_degradation_with_strong_feature_drift",
            severity="high",
            summary="Measured association.",
            evidence_ids=(evidence.id,),
            degraded_metrics=("rmse",),
        )
        hypothesis = self.engine.generate(diagnosis, [evidence])[0]

        self.assertIsInstance(hypothesis, Hypothesis)
        self.assertEqual(hypothesis.type, "covariate_shift")
        self.assertEqual(hypothesis.evidence_ids, [evidence.id])
        self.assertIsNone(hypothesis.confidence)
        self.assertTrue(hypothesis.testable)
        self.assertEqual(hypothesis.recommended_experiment, "retrain_recent_data")

    def test_hypotheses_follow_ranked_evidence_order(self):
        first = _evidence("rank-1", "feature_drift", feature="x")
        second = _evidence("rank-2", "feature_drift", feature="y")
        diagnosis = Diagnosis(
            "diagnosis-1",
            "performance_degradation_with_strong_feature_drift",
            "high",
            "Measured association.",
            evidence_ids=(second.id, first.id),
        )
        hypothesis = self.engine.generate(diagnosis, [first, second])[0]

        self.assertEqual(hypothesis.evidence_ids, [first.id, second.id])
        self.assertEqual(hypothesis.metadata["features"], ["x", "y"])

    def test_non_exploratory_hypothesis_requires_available_evidence(self):
        diagnosis = Diagnosis(
            "diagnosis-1",
            "performance_degradation_with_target_drift",
            "high",
            "Measured association.",
            evidence_ids=("missing",),
        )
        self.assertEqual(self.engine.generate(diagnosis, []), [])

    def test_evidence_free_hypothesis_is_explicitly_exploratory(self):
        diagnosis = Diagnosis(
            "diagnosis-1",
            "degradation_without_observed_covariate_drift",
            "medium",
            "Further investigation is warranted.",
            degraded_metrics=("rmse",),
        )
        hypothesis = self.engine.generate(diagnosis, [])[0]

        self.assertEqual(hypothesis.evidence_ids, [])
        self.assertTrue(hypothesis.metadata["exploratory"])
        self.assertIsNone(hypothesis.confidence)

    def test_major_diagnoses_map_to_ordered_testable_hypotheses(self):
        segment = _evidence(
            "segment-1", "segment_degradation", feature="region", segment="region=north"
        )
        target = _evidence("target-1", "target_drift")
        diagnoses = [
            Diagnosis(
                "segment-diagnosis",
                "performance_degradation_concentrated_in_segments",
                "high",
                "Segment association.",
                evidence_ids=(segment.id,),
            ),
            Diagnosis(
                "target-diagnosis",
                "performance_degradation_with_target_drift",
                "high",
                "Target association.",
                evidence_ids=(target.id,),
            ),
        ]
        hypotheses = self.engine.generate_many(diagnoses, [target, segment])

        self.assertEqual(
            [item.type for item in hypotheses],
            ["target_relationship_shift", "segment_specific_degradation"],
        )
        self.assertTrue(all(item.testable for item in hypotheses))

    def test_research_engine_serializes_typed_traceable_hypothesis(self):
        evidence = _evidence("feature-1", "feature_drift", feature="income")
        diagnosis = Diagnosis(
            "diagnosis-1",
            "performance_degradation_with_strong_feature_drift",
            "high",
            "Measured association.",
            evidence_ids=(evidence.id,),
        )
        result = ResearchEngine().investigate(
            {
                "summary": "One diagnostic pattern.",
                "performance_degradation": [{"metric": "rmse"}],
            },
            [],
            diagnoses=[diagnosis],
            ranked_evidence=[evidence],
        )

        self.assertEqual(result["hypotheses"][0]["evidence_ids"], [evidence.id])
        self.assertEqual(
            result["hypotheses"][0]["recommended_experiment"],
            "retrain_recent_data",
        )
        self.assertEqual(
            result["recommended_next_experiment"]["name"],
            "recent_data_retraining",
        )

    def test_research_loop_consumes_typed_recommended_experiment(self):
        research = {
            "recommended_next_experiment": {"name": "model_search"},
            "hypotheses": [
                {
                    "recommended_experiment": "model_search",
                    "metadata": {"features": []},
                }
            ],
        }
        plan = ResearchLoop._deterministic_plan(
            research, [], ["retrain_recent_data", "model_search"]
        )

        self.assertEqual(plan["action"], "model_search")


if __name__ == "__main__":
    unittest.main()
