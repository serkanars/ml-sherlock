"""Provider-agnostic, constrained LLM planning for ML investigations."""

import json
import logging
import os
from typing import Protocol
from urllib.request import Request, urlopen

from ..config import LLMConfig

logger = logging.getLogger("ml_sherlock.llm")


class LLMProvider(Protocol):
    def plan(self, evidence: dict, history: list[dict], allowed_actions: list[str]) -> dict: ...


class _ConstrainedProvider:
    def _prompt(self, evidence, history, allowed_actions):
        return (
            "You are an ML investigation planner. Statistical evidence is authoritative. "
            "Choose exactly one permitted experiment action; do not invent an action, execute code, "
            "or claim causality. Return exactly one JSON object and no Markdown. Required string fields: "
            "action, hypothesis, rationale. Optional field: confidence.\n"
            f"Allowed actions: {json.dumps(allowed_actions)}\n"
            f"Evidence: {json.dumps(evidence, default=str)}\n"
            f"Previous experiments: {json.dumps(history[-10:], default=str)}"
        )

    @staticmethod
    def _validate(plan, allowed_actions):
        if not isinstance(plan, dict) or plan.get("action") not in allowed_actions:
            raise ValueError("LLM returned an invalid or disallowed experiment action.")
        for field in ("hypothesis", "rationale"):
            if not isinstance(plan.get(field), str) or not plan[field].strip():
                raise ValueError(f"LLM plan must include non-empty '{field}'.")
        plan["source"] = "llm"
        return plan

    @staticmethod
    def _parse_json_object(content):
        """Handle fenced JSON from providers that lack strict structured output."""
        if content is not None and not isinstance(content, str):
            raise ValueError("Provider final content must be a string.")
        text = (content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
            text = text.rsplit("```", 1)[0].strip()
        if not text:
            raise ValueError("Provider returned an empty final content field.")
        try:
            plan = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"Provider did not return a JSON object (content starts: {text[:160]!r}).")
            plan = json.loads(text[start:end + 1])
        if not isinstance(plan, dict):
            raise ValueError("Provider must return a JSON object.")
        return plan


class OllamaProvider(_ConstrainedProvider):
    def __init__(self, config: LLMConfig):
        self.config = config

    def plan(self, evidence, history, allowed_actions):
        payload = {
            "model": self.config.model,
            "stream": False,
            "options": {"temperature": self.config.temperature},
            "messages": [{"role": "user", "content": self._prompt(evidence, history, allowed_actions)}],
        }
        # Ollama Cloud does not support structured outputs. Cloud aliases use
        # the ``-cloud`` suffix, so rely on prompt-constrained JSON there.
        is_cloud_model = self.config.model.endswith("-cloud")
        if not is_cloud_model:
            payload["format"] = "json"
        payload = json.dumps(payload).encode()
        base_url = (self.config.base_url or "http://localhost:11434").rstrip("/")
        logger.info("Ollama plan request | endpoint=%s | model=%s | history=%d | structured_output=%s", base_url, self.config.model, len(history), not is_cloud_model)
        request = Request(f"{base_url}/api/chat", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:  # nosec B310: user-configured provider
                result = json.loads(response.read())
            content = result.get("message", {}).get("content", "")
            plan = self._validate(self._parse_json_object(content), allowed_actions)
            logger.info("Ollama plan received | action=%s | confidence=%s", plan["action"], plan.get("confidence", "unspecified"))
            return plan
        except Exception as exc:
            logger.warning("Ollama planning failed | endpoint=%s | model=%s | reason=%s", base_url, self.config.model, exc)
            raise


class OpenAICompatibleProvider(_ConstrainedProvider):
    def __init__(self, config: LLMConfig):
        self.config = config

    def plan(self, evidence, history, allowed_actions):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install OpenAI provider support with: pip install -e '.[openai]'") from exc
        api_key = os.getenv(self.config.api_key_env or "OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Set the API key environment variable '{self.config.api_key_env or 'OPENAI_API_KEY'}'.")
        client = OpenAI(api_key=api_key, base_url=self.config.base_url)
        logger.info("OpenAI-compatible plan request | endpoint=%s | model=%s | history=%d", self.config.base_url or "default", self.config.model, len(history))
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": self._prompt(evidence, history, allowed_actions)}],
        )
        plan = self._validate(json.loads(response.choices[0].message.content), allowed_actions)
        logger.info("OpenAI-compatible plan received | action=%s | confidence=%s", plan["action"], plan.get("confidence", "unspecified"))
        return plan


def create_provider(config: LLMConfig | None) -> LLMProvider | None:
    if config is None:
        logger.info("LLM planner disabled; deterministic research planning will be used.")
        return None
    provider = config.provider.lower()
    if provider == "ollama":
        logger.info("LLM planner configured | provider=ollama | endpoint=%s | model=%s", config.base_url or "http://localhost:11434", config.model)
        return OllamaProvider(config)
    if provider in {"openai", "openai_compatible"}:
        logger.info("LLM planner configured | provider=%s | endpoint=%s | model=%s", provider, config.base_url or "default", config.model)
        return OpenAICompatibleProvider(config)
    raise ValueError("llm.provider must be ollama, openai, or openai_compatible.")
