"""Configuration-first facade for complete ML-Sherlock investigations."""

import logging
from pathlib import Path
import warnings

from .config import SherlockConfig
from .core.research import ResearchRunner


logger = logging.getLogger("ml_sherlock")


def _ensure_default_logging():
    """Make direct Python usage observable without overriding host logging."""
    if logger.hasHandlers():
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S"
    ))
    handler._ml_sherlock_default = True
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class Sherlock:
    """Run a complete investigation from one Sherlock YAML configuration."""

    def __init__(self, config: str | Path = "sherlock.yaml"):
        _ensure_default_logging()
        self.config_path = Path(config).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Sherlock config not found: {self.config_path}")
        self.config = SherlockConfig.from_yaml(self.config_path)
        self.result = None
        self.fit_result = None
        self.investigation_result = None
        logger.info("Sherlock initialized | config=%s", self.config_path)

    def investigate(self) -> dict:
        logger.info("Investigation started | config=%s", self.config_path)
        mlflow_logger = logging.getLogger("mlflow")
        previous_level = mlflow_logger.level
        mlflow_logger.setLevel(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted in multiple ways.*")
                warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column.*")
                warnings.filterwarnings("ignore", message="`artifact_path` is deprecated.*")
                warnings.filterwarnings("ignore", message="Saving scikit-learn models in the pickle.*")
                warnings.filterwarnings("ignore", message="Encountered an unexpected error while inferring pip requirements.*")
                self.result = ResearchRunner.run_config(self.config)
        except Exception:
            logger.exception("Investigation failed | config=%s", self.config_path)
            raise
        finally:
            mlflow_logger.setLevel(previous_level)
        self.fit_result = self.result["fit"]
        self.investigation_result = self.result["investigation"]
        research = self.investigation_result.get("research", {})
        logger.info(
            "Investigation completed | experiments=%s | report=%s",
            research.get("iterations_completed", "n/a"),
            self.investigation_result.get("report", "n/a"),
        )
        return self.result
