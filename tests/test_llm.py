import json
import unittest
from unittest.mock import MagicMock, patch

from autoresearch.investigation.engine import ResearchEngine
from autoresearch.llm.providers import LLMConfig, OllamaProvider


class OllamaPlanningTests(unittest.TestCase):
    def setUp(self):
        self.action = "retrain_recent_data"
        self.plan = {
            "action": self.action,
            "hypothesis": "Recent data may improve accuracy.",
            "rationale": "Production error increased alongside feature drift.",
        }

    def test_request_is_an_object_and_response_is_validated(self):
        for model in ("qwen3:8b", "example-cloud"):
            for wrapper in ("{}", "```json\n{}\n```", "Selected plan:\n{}"):
                with self.subTest(model=model, wrapper=wrapper):
                    provider = OllamaProvider(LLMConfig(provider="ollama", model=model))
                    response = MagicMock()
                    response.__enter__.return_value.read.return_value = json.dumps({
                        "message": {"content": wrapper.format(json.dumps(self.plan))}
                    }).encode()
                    with patch("autoresearch.llm.providers.urlopen", return_value=response) as send:
                        result = provider.plan({}, [], [self.action])
                    request = send.call_args.args[0]
                    payload = json.loads(request.data)
                    self.assertIsInstance(payload, dict)
                    self.assertEqual(payload["model"], model)
                    self.assertFalse(payload["stream"])
                    self.assertEqual("format" in payload, not model.endswith("-cloud"))
                    self.assertEqual(result["action"], self.action)
                    self.assertEqual(result["source"], "llm")

    def test_invalid_content_is_rejected(self):
        for content in (None, "", "not JSON", "{broken}", "[]", "null", {}, "{} {}"):
            with self.subTest(content=content), self.assertRaises(ValueError):
                OllamaProvider._parse_json_object(content)

    def test_invalid_plans_are_rejected(self):
        for overrides in ({"action": "execute_code"}, {"hypothesis": ""}, {"rationale": None}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                OllamaProvider._validate(self.plan | overrides, [self.action])

    def test_malformed_response_falls_back_to_deterministic_planning(self):
        provider = OllamaProvider(LLMConfig(provider="ollama", model="qwen3:8b"))
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"message":{"content":"invalid"}}'
        diagnosis = {
            "performance_degradation": [{"metric": "rmse"}],
            "summary": "Production error increased.",
        }
        with patch("autoresearch.llm.providers.urlopen", return_value=response):
            result = ResearchEngine(planner=provider).investigate(
                diagnosis, [{"feature": "x", "drift": True}]
            )
        self.assertIn("llm_plan_error", result)
        self.assertNotIn("llm_plan", result)
        self.assertEqual(result["recommended_next_experiment"]["name"], "recent_data_retraining")
