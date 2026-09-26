"""Deterministic evidence-to-hypothesis research layer."""

import logging

from ..evidence import Evidence, make_evidence_id
from .diagnosis import Diagnosis
from .hypotheses import HypothesisEngine

logger = logging.getLogger("ml_sherlock.research")


class ResearchEngine:
    def __init__(self, planner=None, allowed_actions=None, hypothesis_engine=None):
        self.planner = planner
        self.allowed_actions = allowed_actions or [
            "retrain_recent_data",
            "drop_drifted_features",
            "model_search",
        ]
        self.hypothesis_engine = hypothesis_engine or HypothesisEngine()

    def investigate(
        self,
        diagnosis: dict,
        drift: list[dict],
        history=None,
        *,
        diagnoses: list[Diagnosis] | None = None,
        ranked_evidence: list[Evidence] | None = None,
    ) -> dict:
        if diagnoses is None or ranked_evidence is None:
            diagnoses, ranked_evidence = _legacy_typed_inputs(diagnosis, drift)
        typed_hypotheses = self.hypothesis_engine.generate_many(
            diagnoses, ranked_evidence
        )
        hypotheses = [item.to_dict() for item in typed_hypotheses]
        experiment_name = (
            typed_hypotheses[0].recommended_experiment if typed_hypotheses else None
        )
        result = {
            "research_status": "hypotheses_ready",
            "evidence_summary": diagnosis["summary"],
            "hypotheses": hypotheses,
            "recommended_next_experiment": _experiment(experiment_name),
        }
        if self.planner:
            evidence = {
                "diagnosis": diagnosis,
                "drift": drift,
                "deterministic_hypotheses": hypotheses,
            }
            try:
                result["llm_plan"] = self.planner.plan(
                    evidence, history or [], self.allowed_actions
                )
            except Exception as exc:
                result["llm_plan_error"] = str(exc)
                logger.warning(
                    "LLM plan unavailable; using deterministic fallback | reason=%s", exc
                )
        return result


def _legacy_typed_inputs(diagnosis, drift):
    drifted = [item for item in drift if item.get("drift")]
    evidence = []
    for item in drifted:
        feature = item["feature"]
        evidence.append(
            Evidence(
                id=make_evidence_id(
                    "feature_drift", "distribution_effect_size", feature=feature
                ),
                type="feature_drift",
                metric="distribution_effect_size",
                value=item.get("effect_size", item.get("statistic")),
                feature=feature,
                severity=item.get("severity", "medium"),
                metadata={**item, "drift": True},
            )
        )
    degraded = diagnosis.get("performance_degradation", [])
    metrics = tuple(item["metric"] for item in degraded)
    if degraded and evidence:
        pattern = "performance_degradation_with_strong_feature_drift"
        diagnosis_id = "performance_with_feature_drift"
    elif degraded:
        pattern = "degradation_without_observed_covariate_drift"
        diagnosis_id = "degradation_without_covariate_drift"
    elif evidence:
        pattern = "drift_without_measured_performance_degradation"
        diagnosis_id = "drift_without_performance_degradation"
    else:
        pattern = "no_actionable_pattern"
        diagnosis_id = "no_actionable_pattern"
    typed = Diagnosis(
        id=diagnosis_id,
        pattern=pattern,
        severity="medium" if pattern != "no_actionable_pattern" else "info",
        summary=diagnosis.get("summary", "Statistical investigation summary."),
        evidence_ids=tuple(item.id for item in evidence),
        degraded_metrics=metrics,
    )
    return [typed], evidence


def _experiment(name):
    experiments = {
        "retrain_recent_data": {
            "name": "recent_data_retraining",
            "goal": "Test whether representative recent data improves holdout performance.",
            "success_criterion": "Improve the selected metric versus the current champion.",
        },
        "model_search": {
            "name": "model_search",
            "goal": "Compare supported model families on the same fixed holdout.",
            "success_criterion": "Improve the selected metric versus the current champion.",
        },
        "drift_watch": {
            "name": "drift_watch",
            "goal": "Monitor the ranked evidence on the next labelled production window.",
            "success_criterion": "Escalate if measured performance also degrades.",
        },
    }
    return experiments.get(
        name,
        {
            "name": "scheduled_monitoring",
            "goal": "Repeat the comparison on the next labelled production window.",
            "success_criterion": "Keep metrics within the agreed baseline tolerance.",
        },
    )
