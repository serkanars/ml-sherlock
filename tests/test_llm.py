import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ml_sherlock.evidence import Evidence
from ml_sherlock.investigation import Diagnosis
from ml_sherlock.investigation.engine import ResearchEngine
from ml_sherlock.llm.providers import (
    LLMConfig,
    OllamaProvider,
    OpenAICompatibleProvider,
)


class LLMPlanningTests(unittest.TestCase):
    def setUp(self):
        self.action = "feature_subset_search"
        self.context = {
            "performance_summary": {
                "status": "degraded",
                "summary": "One metric degraded.",
                "degraded_metrics": [{
                    "metric": "rmse", "baseline": 1.0,
                    "production": 1.4, "change_pct": 40.0,
                }],
            },
            "diagnosis": [{
                "id": "diagnosis-1",
                "pattern": "performance_degradation_with_strong_feature_drift",
                "severity": "high",
                "summary": "Measured association.",
                "evidence_ids": ["evidence-1"],
                "degraded_metrics": ["rmse"],
            }],
            "top_evidence": [
                {
                    "id": "evidence-1", "type": "feature_drift",
                    "metric": "psi", "value": .4, "feature": "x",
                    "severity": "high", "metadata": {},
                },
                {
                    "id": "evidence-2", "type": "target_drift",
                    "metric": "psi", "value": .3, "severity": "medium",
                    "metadata": {},
                },
            ],
            "hypotheses": [{
                "id": "covariate_shift",
                "type": "covariate_shift",
                "claim": "Measured feature drift is associated with degradation.",
                "evidence_ids": ["evidence-1"],
                "confidence": None,
                "testable": True,
                "recommended_experiment": self.action,
                "metadata": {},
            }],
        }
        self.plan = {
            "hypothesis_id": "covariate_shift",
            "evidence_ids": ["evidence-1"],
            "action": self.action,
            "rationale": "The measured evidence supports testing this bounded action.",
        }

    def test_ollama_request_uses_structured_context_and_validates_response(self):
        for model in ("qwen3:8b", "example-cloud"):
            for wrapper in ("{}", "```json\n{}\n```", "Selected plan:\n{}"):
                with self.subTest(model=model, wrapper=wrapper):
                    provider = OllamaProvider(
                        LLMConfig(provider="ollama", model=model)
                    )
                    response = MagicMock()
                    response.__enter__.return_value.read.return_value = json.dumps({
                        "message": {"content": wrapper.format(json.dumps(self.plan))}
                    }).encode()
                    with patch(
                        "ml_sherlock.llm.providers.urlopen", return_value=response
                    ) as send:
                        result = provider.plan(
                            self.context, [{"action": "model_search"}], [self.action]
                        )
                    request = send.call_args.args[0]
                    payload = json.loads(request.data)
                    prompt = payload["messages"][0]["content"]
                    self.assertIsInstance(payload, dict)
                    self.assertEqual(payload["model"], model)
                    self.assertFalse(payload["stream"])
                    self.assertEqual("format" in payload, not model.endswith("-cloud"))
                    self.assertIn('"performance_summary"', prompt)
                    self.assertIn('"top_evidence"', prompt)
                    self.assertIn('"hypotheses"', prompt)
                    self.assertIn("do not calculate statistics", prompt.lower())
                    self.assertEqual(result["hypothesis_id"], "covariate_shift")
                    self.assertEqual(result["evidence_ids"], ["evidence-1"])
                    self.assertEqual(result["source"], "llm")

    def test_openai_compatible_provider_uses_the_same_validation(self):
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(self.plan))
            )]
        )
        openai_module = SimpleNamespace(OpenAI=MagicMock(return_value=client))
        config = LLMConfig(
            provider="openai_compatible", model="test-model",
            api_key_env="TEST_LLM_KEY",
        )
        with patch.dict(sys.modules, {"openai": openai_module}), patch.dict(
            os.environ, {"TEST_LLM_KEY": "secret"}
        ):
            result = OpenAICompatibleProvider(config).plan(
                self.context, [], [self.action]
            )

        self.assertEqual(result["evidence_ids"], ["evidence-1"])
        prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn('"diagnosis"', prompt)
        self.assertNotIn("raw_dataframe", prompt)

    def test_invalid_content_is_rejected(self):
        for content in (None, "", "not JSON", "{broken}", "[]", "null", {}, "{} {}"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                OllamaProvider._parse_json_object(content)

    def test_invalid_actions_hypotheses_and_evidence_are_rejected(self):
        invalid = (
            {"action": "execute_code"},
            {"hypothesis_id": "invented-hypothesis"},
            {"evidence_ids": ["hallucinated-evidence"]},
            {"evidence_ids": ["evidence-2"]},
            {"evidence_ids": []},
            {"rationale": ""},
            {"unexpected": "field"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                OllamaProvider._validate(
                    self.plan | overrides, [self.action], self.context
                )

    def test_engine_sends_only_compact_ranked_evidence_context(self):
        evidence = [
            Evidence(
                id=f"evidence-{index}", type="feature_drift", metric="psi",
                value=1 - index / 100, feature=f"x{index}", severity="high",
                metadata={"effect_size": 1 - index / 100, "raw_bins": list(range(50))},
            )
            for index in range(12)
        ]
        diagnosis = Diagnosis(
            id="diagnosis-1",
            pattern="performance_degradation_with_strong_feature_drift",
            severity="high",
            summary="Measured feature drift is associated with degradation.",
            evidence_ids=tuple(item.id for item in evidence),
            degraded_metrics=("rmse",),
        )
        planner = MagicMock()
        planner.plan.return_value = self.plan
        summary = {
            "status": "degraded",
            "summary": "One metric degraded; twelve features drifted.",
            "performance_degradation": [{
                "metric": "rmse", "baseline": 1.0,
                "production": 1.4, "change_pct": 40.0,
            }],
            "drifted_features": [{"feature": "x0", "large_raw_payload": [1] * 100}],
        }

        result = ResearchEngine(planner=planner, top_evidence_limit=10).investigate(
            summary,
            [{"feature": "x0", "raw_values": [1, 2, 3]}],
            history=[{"iteration": 1, "action": "model_search"}],
            diagnoses=[diagnosis],
            ranked_evidence=evidence,
        )

        context, history, allowed = planner.plan.call_args.args
        self.assertEqual(
            set(context),
            {"performance_summary", "diagnosis", "top_evidence", "hypotheses"},
        )
        self.assertEqual(len(context["top_evidence"]), 10)
        self.assertEqual(context["top_evidence"][0]["id"], "evidence-0")
        self.assertNotIn("raw_bins", context["top_evidence"][0]["metadata"])
        self.assertNotIn("drifted_features", context["performance_summary"])
        self.assertEqual(history[0]["action"], "model_search")
        self.assertIn(self.action, allowed)
        self.assertEqual(result["llm_plan"], self.plan)

    def test_hallucinated_evidence_falls_back_to_deterministic_planning(self):
        provider = OllamaProvider(LLMConfig(provider="ollama", model="qwen3:8b"))
        invalid_plan = self.plan | {"evidence_ids": ["invented-evidence"]}
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "message": {"content": json.dumps(invalid_plan)}
        }).encode()
        evidence = Evidence(
            id="evidence-1", type="feature_drift", metric="psi", value=.5,
            feature="x", severity="high", metadata={"drift": True},
        )
        diagnosis = Diagnosis(
            id="diagnosis-1",
            pattern="performance_degradation_with_strong_feature_drift",
            severity="high",
            summary="Measured association.",
            evidence_ids=(evidence.id,),
            degraded_metrics=("rmse",),
        )
        summary = {
            "status": "degraded",
            "performance_degradation": [{"metric": "rmse"}],
            "summary": "Production error increased.",
        }
        with patch("ml_sherlock.llm.providers.urlopen", return_value=response):
            result = ResearchEngine(planner=provider).investigate(
                summary, [], diagnoses=[diagnosis], ranked_evidence=[evidence]
            )

        self.assertIn("unknown evidence IDs", result["llm_plan_error"])
        self.assertNotIn("llm_plan", result)
        self.assertEqual(
            result["recommended_next_experiment"]["name"],
            "feature_subset_search",
        )


if __name__ == "__main__":
    unittest.main()
