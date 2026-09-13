"""Deterministic evidence-to-hypothesis research layer.

This module intentionally does not use an LLM: hypotheses are traceable to the
measured performance and drift evidence that produced them.
"""

import logging

logger = logging.getLogger("ml_sherlock.research")


class ResearchEngine:
    def __init__(self, planner=None, allowed_actions=None):
        self.planner = planner
        self.allowed_actions = allowed_actions or ["retrain_recent_data", "drop_drifted_features", "model_search"]

    def investigate(self, diagnosis: dict, drift: list[dict], history=None) -> dict:
        degraded = diagnosis["performance_degradation"]
        drifted = [item for item in drift if item["drift"]]
        hypotheses = []

        if degraded and drifted:
            features = [item["feature"] for item in drifted]
            hypotheses.append({
                "id": "covariate_shift",
                "title": "Covariate shift is contributing to model degradation",
                "confidence": "high",
                "claim": "Production feature distributions differ materially from the training reference while predictive performance worsened.",
                "evidence": {
                    "degraded_metrics": [item["metric"] for item in degraded],
                    "drifted_features": features,
                },
                "experiment": {
                    "name": "recent_data_retraining",
                    "goal": "Test whether training with representative recent data restores production performance.",
                    "success_criterion": "Candidate model improves the selected production metric versus the baseline.",
                },
            })

        if degraded and not drifted:
            hypotheses.append({
                "id": "unobserved_or_concept_drift",
                "title": "Performance degradation is not explained by observed feature drift",
                "confidence": "medium",
                "claim": "The measured input features are stable, so investigate target/concept drift, labels, and untracked inputs.",
                "evidence": {"degraded_metrics": [item["metric"] for item in degraded], "drifted_features": []},
                "experiment": {
                    "name": "residual_and_target_analysis",
                    "goal": "Compare target and residual distributions across production segments.",
                    "success_criterion": "Identify a segment or target shift that explains elevated error.",
                },
            })

        if not degraded and drifted:
            hypotheses.append({
                "id": "benign_input_drift",
                "title": "Input drift has not yet affected measured performance",
                "confidence": "medium",
                "claim": "Feature distributions changed, but the current labelled production sample does not show metric deterioration.",
                "evidence": {"degraded_metrics": [], "drifted_features": [item["feature"] for item in drifted]},
                "experiment": {
                    "name": "drift_watch",
                    "goal": "Monitor the drifted features and collect the next labelled production window.",
                    "success_criterion": "Escalate only if the selected metric crosses its agreed threshold.",
                },
            })

        if not hypotheses:
            hypotheses.append({
                "id": "no_actionable_anomaly",
                "title": "No actionable degradation in the current comparison",
                "confidence": "high",
                "claim": "Neither the measured metrics nor feature distributions require an investigation at this time.",
                "evidence": {"degraded_metrics": [], "drifted_features": []},
                "experiment": {
                    "name": "scheduled_monitoring",
                    "goal": "Repeat the comparison on the next labelled production window.",
                    "success_criterion": "Maintain metrics within the agreed baseline tolerance.",
                },
            })

        result = {
            "research_status": "hypotheses_ready",
            "evidence_summary": diagnosis["summary"],
            "hypotheses": hypotheses,
            "recommended_next_experiment": hypotheses[0]["experiment"],
        }
        if self.planner:
            evidence = {"diagnosis": diagnosis, "drift": drift, "deterministic_hypotheses": hypotheses}
            try:
                result["llm_plan"] = self.planner.plan(evidence, history or [], self.allowed_actions)
            except Exception as exc:
                result["llm_plan_error"] = str(exc)
                logger.warning("LLM plan unavailable; using deterministic fallback | reason=%s", exc)
        return result
