import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml_sherlock.evidence import Evidence
from ml_sherlock.investigation.loop import ResearchLoop


class CumulativeLoopTests(unittest.TestCase):
    def test_deterministic_planner_balances_supported_drift_actions(self):
        research = {
            "recommended_next_experiment": {"name": "recent_data_retraining"},
            "hypotheses": [{"evidence": {"drifted_features": ["x"]}}],
        }
        history = []
        actions = []
        for iteration in range(6):
            plan = ResearchLoop._deterministic_plan(
                research, history,
                ["retrain_recent_data", "drop_drifted_features", "model_search"],
            )
            actions.append(plan["action"])
            history.append({"iteration": iteration + 1, "action": plan["action"]})
        self.assertEqual(actions, [
            "retrain_recent_data", "model_search", "drop_drifted_features",
            "retrain_recent_data", "model_search", "drop_drifted_features",
        ])

    def test_only_accepted_models_and_features_are_carried_forward(self):
        models = [SimpleNamespace(named_steps={"model": RandomForestRegressor(random_state=i)}) for i in range(4)]
        runner = Mock()
        runner.selection_metric = "rmse"
        runner.trainer.model_name.return_value = "random_forest"
        runner.run_action.side_effect = [
            self.result(models[1], "validated", ["x"], 20, 8),
            self.result(models[2], "rejected", ["x"], -10, 9),
            self.result(models[3], "validated", ["x"], 2, 7.84),
        ]
        engine = Mock()
        engine.investigate.side_effect = lambda *args: {"llm_plan": {"action": "retrain_recent_data"}}
        data = pd.DataFrame({"x": [1, 2], "z": [3, 4], "target": [5, 6]})
        result = ResearchLoop(engine, runner, max_experiments=3).run(
            data, data, "target", models[0], {}, []
        )
        calls = runner.run_action.call_args_list
        self.assertIs(calls[0].args[4], models[0])
        self.assertIs(calls[1].args[4], models[1])
        self.assertIs(calls[2].args[4], models[1])
        self.assertEqual(list(calls[2].args[1].columns), ["x", "target"])
        self.assertEqual([item["parent_iteration"] for item in result["experiments"]], [0, 1, 1])
        self.assertTrue(all("hypothesis_id" in item for item in result["experiments"]))
        self.assertIs(result["_champion"], models[3])
        self.assertEqual(result["decision"]["iteration"], 3)
        self.assertEqual(result["decision"]["parameters"]["random_state"], 3)

    @staticmethod
    def result(model, status, features, improvement, score):
        return {"_model": model, "status": status, "used_features": features,
                "improvement_pct": improvement, "candidate_model": "random_forest",
                "recommended_model": "random_forest", "selection_metric": "rmse",
                "training_duration_seconds": 0, "action": "retrain_recent_data",
                "candidate_metrics": {"rmse": score}, "baseline_metrics": {"rmse": 10}}

    def test_final_evaluation_can_block_deployment_without_reselecting(self):
        baseline = SimpleNamespace(named_steps={"model": RandomForestRegressor(random_state=0)})
        candidate = SimpleNamespace(named_steps={"model": RandomForestRegressor(random_state=1)})
        runner = Mock()
        runner.selection_metric = "rmse"
        runner.min_improvement_pct = 1
        runner.trainer.model_name.return_value = "random_forest"
        runner.trainer.evaluate.side_effect = [{"rmse": 10}, {"rmse": 12}]
        runner.run_action.return_value = self.result(candidate, "validated", ["x"], 20, 8)
        engine = Mock()
        engine.investigate.return_value = {"llm_plan": {"action": "retrain_recent_data"}}
        development = pd.DataFrame({"x": [1, 2], "target": [3, 4]})
        final = pd.DataFrame({"x": [100, 200], "target": [300, 400]})
        result = ResearchLoop(engine, runner, max_experiments=1).run(
            development, development, "target", baseline, {}, [], final
        )
        self.assertEqual(result["decision"]["deployment_status"], "keep_baseline")
        self.assertEqual(result["decision"]["final_improvement_pct"], -20)
        self.assertIs(result["_champion"], candidate)
        pd.testing.assert_frame_equal(runner.run_action.call_args.args[2], development)
        pd.testing.assert_frame_equal(runner.trainer.evaluate.call_args_list[1].args[1], final)

    def test_llm_cannot_execute_an_action_outside_allowed_actions(self):
        baseline = SimpleNamespace(
            named_steps={"model": RandomForestRegressor(random_state=0)}
        )
        runner = Mock()
        runner.selection_metric = "rmse"
        runner.trainer.model_name.return_value = "random_forest"
        engine = Mock()
        engine.investigate.return_value = {
            "llm_plan": {"action": "execute_python", "source": "llm"}
        }
        data = pd.DataFrame({"x": [1, 2], "target": [3, 4]})

        result = ResearchLoop(
            engine, runner, max_experiments=1,
            allowed_actions=["retrain_recent_data"],
        ).run(data, data, "target", baseline, {}, [])

        runner.run_action.assert_not_called()
        self.assertEqual(result["history"][0]["reason"], "disallowed_action")
        self.assertEqual(result["experiments"], [])

    def test_hypothesis_evidence_is_forwarded_to_evidence_driven_action(self):
        baseline = SimpleNamespace(
            named_steps={"model": RandomForestRegressor(random_state=0)}
        )
        candidate = SimpleNamespace(
            named_steps={"model": RandomForestRegressor(random_state=1)}
        )
        evidence = Evidence(
            id="segment-b", type="segment_degradation",
            metric="rmse_degradation_pct", value=80.0,
            feature="group", segment="group=b",
            metadata={"definition": {"category": "b"}},
        )
        runner = Mock()
        runner.selection_metric = "rmse"
        runner.trainer.model_name.return_value = "random_forest"
        runner.run_action.return_value = self.result(
            candidate, "rejected", ["x", "group"], -1, 10.1
        )
        engine = Mock()
        engine.investigate.return_value = {
            "llm_plan": {
                "action": "segment_retraining",
                "hypothesis_id": "segment_specific_degradation",
            },
            "hypotheses": [{
                "id": "segment_specific_degradation",
                "recommended_experiment": "segment_retraining",
                "evidence_ids": [evidence.id],
            }],
        }
        data = pd.DataFrame({
            "x": [1, 2], "group": ["a", "b"], "target": [3, 4]
        })

        ResearchLoop(
            engine, runner, max_experiments=1,
            allowed_actions=["segment_retraining"],
        ).run(
            data, data, "target", baseline, {}, [],
            diagnoses=[], ranked_evidence=[evidence],
        )

        kwargs = runner.run_action.call_args.kwargs
        self.assertEqual(kwargs["hypothesis_id"], "segment_specific_degradation")
        self.assertEqual(kwargs["evidence_ids"], [evidence.id])
        self.assertEqual(kwargs["evidence"], [evidence])
