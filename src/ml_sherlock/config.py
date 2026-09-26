"""Typed, versioned configuration for ML-Sherlock investigations."""

from pathlib import Path
from typing import Literal
import warnings

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml


ModelName = Literal["random_forest", "extra_trees", "xgboost", "lightgbm"]
MetricName = Literal["rmse", "mae", "mape", "r2"]
ExperimentAction = Literal["retrain_recent_data", "drop_drifted_features", "model_search"]
ProviderName = Literal["ollama", "openai", "openai_compatible"]
SUPPORTED_MODELS = ("random_forest", "extra_trees", "xgboost", "lightgbm")
SUPPORTED_ACTIONS = ("retrain_recent_data", "drop_drifted_features", "model_search")


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DataConfig(StrictConfig):
    target: str = Field(min_length=1)
    train: Path
    production: Path

    @field_validator("target")
    @classmethod
    def target_must_not_be_blank(cls, value):
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class TrackingConfig(StrictConfig):
    uri: str = "sqlite:///artifacts/mlflow.db"
    experiment: str = "ml-sherlock"


class ModelConfig(StrictConfig):
    candidates: list[ModelName] = Field(default_factory=lambda: list(SUPPORTED_MODELS), min_length=1)
    selection_metric: MetricName = "rmse"
    random_state: int = 42

    @field_validator("candidates")
    @classmethod
    def candidates_must_be_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("model candidates must be unique")
        return value


class DriftConfig(StrictConfig):
    enabled: bool = True
    p_value_threshold: float = Field(default=0.05, gt=0, lt=1)


class ErrorAnalysisConfig(StrictConfig):
    enabled: bool = True
    metrics: list[MetricName] = Field(
        default_factory=lambda: ["rmse", "mae", "mape", "r2"], min_length=1
    )

    @field_validator("metrics")
    @classmethod
    def metrics_must_be_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("error-analysis metrics must be unique")
        return value


class SegmentConfig(StrictConfig):
    enabled: bool = False
    columns: list[str] = Field(default_factory=list)
    min_rows: int = Field(default=50, ge=2)
    max_segments: int = Field(default=20, ge=1, le=1000)

    @model_validator(mode="after")
    def enabled_segments_require_columns(self):
        if self.enabled and not self.columns:
            raise ValueError("enabled segment analysis requires at least one column")
        return self


class InvestigationConfig(StrictConfig):
    drift: DriftConfig = Field(default_factory=DriftConfig)
    error_analysis: ErrorAnalysisConfig = Field(default_factory=ErrorAnalysisConfig)
    segments: SegmentConfig = Field(default_factory=SegmentConfig)


class ExperimentConfig(StrictConfig):
    max_iterations: int = Field(default=5, ge=1, le=1000)
    adaptation_fraction: float = Field(default=0.5, gt=0, lt=1)
    min_improvement_pct: float = 1.0
    allowed_actions: list[ExperimentAction] = Field(
        default_factory=lambda: list(SUPPORTED_ACTIONS), min_length=1
    )

    @field_validator("allowed_actions")
    @classmethod
    def actions_must_be_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("experiment actions must be unique")
        return value


class LLMConfig(StrictConfig):
    enabled: bool = False
    provider: ProviderName | None = None
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(default=0.1, ge=0, le=2)
    timeout_seconds: int = Field(default=60, ge=1, le=3600)

    @model_validator(mode="after")
    def enabled_provider_requires_identity(self):
        if self.enabled and (not self.provider or not self.model or not self.model.strip()):
            raise ValueError("enabled LLM requires provider and model")
        return self


class ReportConfig(StrictConfig):
    output: Path = Path("artifacts/report.html")


class SherlockConfig(StrictConfig):
    version: Literal[1] = 1
    data: DataConfig
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    investigation: InvestigationConfig = Field(default_factory=InvestigationConfig)
    experiments: ExperimentConfig = Field(default_factory=ExperimentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SherlockConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        if not isinstance(raw, dict):
            raise ValueError("Sherlock configuration root must be a mapping.")
        if _is_legacy(raw):
            warnings.warn(
                "Legacy Sherlock YAML keys are deprecated; migrate to version: 1 typed configuration.",
                DeprecationWarning,
                stacklevel=2,
            )
            raw = _migrate_legacy(raw)
        config = cls.model_validate(raw)
        base = config_path.parent
        data = config.data.model_copy(update={
            "train": _resolve_path(config.data.train, base),
            "production": _resolve_path(config.data.production, base),
        })
        report = config.report.model_copy(update={"output": _resolve_path(config.report.output, base)})
        tracking = config.tracking.model_copy(update={
            "uri": _resolve_tracking_uri(config.tracking.uri, base)
        })
        return config.model_copy(update={"data": data, "report": report, "tracking": tracking})


def _resolve_path(value: Path, base: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _resolve_tracking_uri(uri: str, base: Path) -> str:
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    location = uri[len(prefix):]
    if location == ":memory:" or Path(location).is_absolute():
        return uri
    return prefix + (base / location).resolve().as_posix()


def _is_legacy(raw: dict) -> bool:
    return "project" in raw or "model" in raw or "research" in raw


def _remap(section, names):
    if not isinstance(section, dict):
        return section
    return {names.get(key, key): value for key, value in section.items()}


def _migrate_legacy(raw: dict) -> dict:
    project = dict(raw.get("project") or {})
    data = _remap(raw.get("data") or {}, {"train_path": "train", "production_path": "production"})
    if "target" in project:
        data["target"] = project.pop("target")
    migrated = {
        "version": 1,
        "data": data,
        "tracking": _remap(raw.get("tracking") or {}, {"experiment_name": "experiment"}),
        "models": _remap(raw.get("model") or {}, {}),
        "llm": raw.get("llm") or {},
        "report": _remap(raw.get("report") or {}, {"output_path": "output"}),
    }
    research = dict(raw.get("research") or {})
    threshold = research.pop("drift_p_value_threshold", None)
    migrated["investigation"] = {
        "drift": {} if threshold is None else {"p_value_threshold": threshold}
    }
    migrated["experiments"] = _remap(research, {"max_experiments": "max_iterations"})
    if project:
        migrated["project"] = project
    for key, value in raw.items():
        if key not in {"project", "data", "tracking", "model", "research", "llm", "report"}:
            migrated[key] = value
    return migrated
