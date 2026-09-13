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

    def run(self, reference, production, target, baseline_model, diagnosis, drift):
        history, experiments = [], []
        last_research = None
        for iteration in range(1, self.max_experiments + 1):
            last_research = self.research_engine.investigate(diagnosis, drift, history)
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
                action, reference, production, target, baseline_model,
                [item["feature"] for item in drift if item["drift"]],
            )
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
            })

        if not last_research:
            last_research = self.research_engine.investigate(diagnosis, drift, history)
        best = self._best_validated(experiments)
        last_research.update({
            "experiments": experiments, "history": history, "iterations_completed": len(experiments),
            "recommendation": self._recommendation(best, experiments),
        })
        return last_research

    @staticmethod
    def _deterministic_plan(research):
        next_experiment = research["recommended_next_experiment"]["name"]
        action = "retrain_recent_data" if next_experiment == "recent_data_retraining" else "stop"
        return {"action": action, "hypothesis": "Deterministic fallback", "rationale": "Derived from statistical evidence.", "source": "deterministic"}

    @staticmethod
    def _best_validated(experiments):
        valid = [item for item in experiments if item["status"] == "validated"]
        return max(valid, key=lambda item: item["improvement_pct"], default=None)

    @staticmethod
    def _recommendation(best, experiments):
        if best:
            return f"Recommend {best['recommended_model']} from iteration {best['iteration']} for deployment review"
        if experiments:
            return "Keep the baseline; no scheduled experiment met the validation criterion."
        return "Continue monitoring; no executable experiment was selected."
