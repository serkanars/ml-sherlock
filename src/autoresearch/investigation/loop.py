"""Bounded research scheduler: planner -> guarded action -> evidence history."""

import logging

logger = logging.getLogger("ml_sherlock.research")


class ResearchLoop:
    def __init__(self, research_engine, experiment_runner, max_experiments=5,
                 allowed_actions=None, random_state=42, iteration_logger=None):
        self.research_engine = research_engine
        self.experiment_runner = experiment_runner
        self.max_experiments = max_experiments
        self.allowed_actions = allowed_actions or ["retrain_recent_data", "drop_drifted_features", "model_search"]
        self.random_state = random_state
        self.iteration_logger = iteration_logger

    def run(self, reference, production, target, baseline_model, diagnosis, drift, final_evaluation=None):
        history, experiments = [], []
        last_research = None
        champion = baseline_model
        champion_iteration = 0
        active_features = [column for column in reference.columns if column != target]
        for iteration in range(1, self.max_experiments + 1):
            evidence = dict(diagnosis, current_model=self.experiment_runner.trainer.model_name(champion),
                            current_iteration=champion_iteration, active_features=list(active_features))
            last_research = self.research_engine.investigate(evidence, drift, history)
            plan = last_research.get("llm_plan") or self._deterministic_plan(last_research)
            action = plan.get("action")
            if action == "stop":
                history.append({"iteration": iteration, "action": "stop", "source": plan.get("source", "deterministic")})
                break
            if action not in self.allowed_actions:
                history.append({"iteration": iteration, "action": action, "status": "rejected", "reason": "disallowed_action"})
                break
            # Keep the production holdout fixed for every iteration. Otherwise
            # improvements from different trials would not be comparable. The
            # training seed varies instead, producing reproducible candidates
            # without moving the evaluation goalposts.
            self.experiment_runner.random_state = self.random_state
            self.experiment_runner.trainer.random_state = self.random_state + iteration
            result = self.experiment_runner.run_action(
                action, reference[active_features + [target]], production[active_features + [target]], target, champion,
                [item["feature"] for item in drift if item["drift"] and item["feature"] in active_features],
            )
            result["parent_iteration"] = champion_iteration
            if result["status"] == "validated":
                champion = result["_model"]
                champion_iteration = iteration
                active_features = result["used_features"]
            result["recommended_model"] = self.experiment_runner.trainer.model_name(champion)
            result["champion_iteration"] = champion_iteration
            result["iteration"] = iteration
            result["training_seed"] = self.experiment_runner.trainer.random_state
            result["planner"] = plan
            result["drifted_features"] = [item["feature"] for item in drift if item["drift"]]
            if self.iteration_logger:
                result["mlflow_run_id"] = self.iteration_logger(result, drift)
            logger.info(
                "[%02d/%02d] action=%s | hypothesis=%s | drift=%s | model=%s | %s=%+.2f%% | train=%.2fs | status=%s",
                iteration, self.max_experiments, action, plan.get("hypothesis", "n/a"),
                ", ".join(result["drifted_features"]) or "none", result["recommended_model"],
                result["selection_metric"], result["improvement_pct"], result["training_duration_seconds"], result["status"],
            )
            experiments.append(result)
            history.append({
                "iteration": iteration, "action": action, "status": result["status"],
                "improvement_pct": result["improvement_pct"], "recommended_model": result["recommended_model"],
                "candidate_model": result["candidate_model"],
                "candidate_metrics": result["candidate_metrics"],
                "baseline_metrics": result["baseline_metrics"],
                "used_features": result["used_features"],
                "champion_iteration": champion_iteration,
                "active_features": list(active_features),
            })

        if not last_research:
            last_research = self.research_engine.investigate(diagnosis, drift, history)
        best = next((item for item in experiments if item["iteration"] == champion_iteration), None)
        trainer = self.experiment_runner.trainer
        decision = {
            "model": trainer.model_name(champion),
            "iteration": champion_iteration,
            "parameters": champion.named_steps["model"].get_params(),
            "used_features": active_features,
            "dropped_features": [c for c in reference.columns if c != target and c not in active_features],
            "selection_metric": self.experiment_runner.selection_metric,
            "selection_metrics": best["candidate_metrics"] if best else (experiments[0]["baseline_metrics"] if experiments else None),
            "drift_evidence": drift,
            "monitoring": [
                "Track feature distributions and missing-value rates against the training reference.",
                "Track labelled production errors on new time windows; investigate drift together with performance changes.",
                "Use representative recent labelled data for retraining and validate before deployment.",
                "Observed drift is an association, not proof of causality or a guarantee of future degradation.",
            ],
            "validation_note": "Experiment holdout is reused for selection. Final evaluation is reserved until selection is complete.",
        }
        if final_evaluation is not None:
            original = trainer.evaluate(baseline_model, final_evaluation, target)
            selected = trainer.evaluate(champion, final_evaluation[active_features + [target]], target)
            metric = self.experiment_runner.selection_metric
            before, after = original[metric], selected[metric]
            improvement = ((after - before) if metric == "r2" else (before - after)) / max(abs(before), 1e-12) * 100
            decision.update(final_baseline_metrics=original, final_candidate_metrics=selected,
                            final_improvement_pct=improvement, final_rows=len(final_evaluation),
                            deployment_status="review_candidate" if champion_iteration and improvement >= self.experiment_runner.min_improvement_pct else "keep_baseline")
        else:
            decision["deployment_status"] = "requires_independent_evaluation"
        last_research.update({
            "experiments": experiments, "history": history, "iterations_completed": len(experiments),
            "recommendation": self._recommendation(best, experiments),
            "decision": decision,
            "_champion": champion,
        })
        if decision["deployment_status"] == "keep_baseline":
            last_research["recommendation"] = "Keep the baseline; the selected candidate did not clear independent final evaluation."
        return last_research

    @staticmethod
    def _deterministic_plan(research):
        next_experiment = research["recommended_next_experiment"]["name"]
        action = "retrain_recent_data" if next_experiment == "recent_data_retraining" else "stop"
        return {"action": action, "hypothesis": "Deterministic fallback", "rationale": "Derived from statistical evidence.", "source": "deterministic"}

    @staticmethod
    def _recommendation(best, experiments):
        if best:
            return f"Recommend {best['recommended_model']} from iteration {best['iteration']} for deployment review"
        if experiments:
            return "Keep the baseline; no scheduled experiment met the validation criterion."
        return "Continue monitoring; no executable experiment was selected."
