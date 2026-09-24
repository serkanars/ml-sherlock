"""YAML configuration loading for reproducible ML-Sherlock investigations."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from .llm import LLMConfig
from .models.trainer import BaselineTrainer


@dataclass(frozen=True)
class InvestigationConfig:
    target: str
    train_path: Path
    production_path: Path
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "ml-sherlock"
    random_state: int = 42
    model_candidates: tuple[str, ...] = BaselineTrainer.SUPPORTED_MODELS
    selection_metric: str = "rmse"
    drift_p_value_threshold: float = 0.05
    adaptation_fraction: float = .5
    min_improvement_pct: float = 1.0
    max_experiments: int = 5
    allowed_actions: tuple[str, ...] = ("retrain_recent_data", "drop_drifted_features", "model_search")
    report_path: Path = Path("autoresearch_report.html")
    llm: LLMConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InvestigationConfig":
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}

        project = raw.get("project", {})
        data = raw.get("data", {})
        tracking = raw.get("tracking", {})
        model = raw.get("model", {})
        research = raw.get("research", {})
        report = raw.get("report", {})
        llm = raw.get("llm", {})

        target = project.get("target")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("'project.target' must be a non-empty string.")
        if not data.get("train_path") or not data.get("production_path"):
            raise ValueError("'data.train_path' and 'data.production_path' are required.")

        def resolve(value: str) -> Path:
            return (config_path.parent / value).resolve()

        threshold = float(research.get("drift_p_value_threshold", 0.05))
        if not 0 < threshold < 1:
            raise ValueError("'research.drift_p_value_threshold' must be between 0 and 1.")
        candidates = model.get("candidates", list(BaselineTrainer.SUPPORTED_MODELS))
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("'model.candidates' must be a non-empty list.")
        invalid_candidates = set(candidates) - set(BaselineTrainer.SUPPORTED_MODELS)
        if invalid_candidates:
            raise ValueError(
                f"'model.candidates' contains unsupported models: {sorted(invalid_candidates)}. "
                f"Supported models: {list(BaselineTrainer.SUPPORTED_MODELS)}."
            )
        metric = model.get("selection_metric", "rmse").lower()
        if metric not in {"rmse", "mae", "mape", "r2"}:
            raise ValueError("'model.selection_metric' must be rmse, mae, mape, or r2.")
        adaptation_fraction = float(research.get("adaptation_fraction", .5))
        if not 0 < adaptation_fraction < 1:
            raise ValueError("'research.adaptation_fraction' must be between 0 and 1.")
        max_experiments = int(research.get("max_experiments", 5))
        if not 1 <= max_experiments <= 1000:
            raise ValueError("'research.max_experiments' must be between 1 and 1000.")
        allowed_actions = research.get("allowed_actions", ["retrain_recent_data", "drop_drifted_features", "model_search"])
        supported_actions = {"retrain_recent_data", "drop_drifted_features", "model_search"}
        if not isinstance(allowed_actions, list) or not allowed_actions or not set(allowed_actions) <= supported_actions:
            raise ValueError(f"'research.allowed_actions' must be a non-empty subset of {sorted(supported_actions)}.")
        llm_config = None
        if llm.get("enabled", False):
            if not llm.get("provider") or not llm.get("model"):
                raise ValueError("Enabled LLM requires 'llm.provider' and 'llm.model'.")
            llm_config = LLMConfig(
                provider=llm["provider"], model=llm["model"], base_url=llm.get("base_url"),
                api_key_env=llm.get("api_key_env"), temperature=float(llm.get("temperature", .1)),
                timeout_seconds=int(llm.get("timeout_seconds", 60)),
            )

        return cls(
            target=target,
            train_path=resolve(data["train_path"]),
            production_path=resolve(data["production_path"]),
            tracking_uri=tracking.get("uri", "sqlite:///mlflow.db"),
            experiment_name=tracking.get("experiment_name", "ml-sherlock"),
            random_state=int(model.get("random_state", 42)),
            model_candidates=tuple(candidates),
            selection_metric=metric,
            drift_p_value_threshold=threshold,
            adaptation_fraction=adaptation_fraction,
            min_improvement_pct=float(research.get("min_improvement_pct", 1.0)),
            max_experiments=max_experiments,
            allowed_actions=tuple(allowed_actions),
            report_path=resolve(report.get("output_path", "autoresearch_report.html")),
            llm=llm_config,
        )
