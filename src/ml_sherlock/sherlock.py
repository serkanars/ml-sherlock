"""Configuration-first facade for complete ML-Sherlock investigations."""

from pathlib import Path

from autoresearch.core.research import AutoResearch


class Sherlock:
    """Run a complete investigation from one Sherlock YAML configuration."""

    def __init__(self, config: str | Path = "sherlock.yaml"):
        self.config_path = Path(config).expanduser().resolve()
        self.result = None
        self.fit_result = None
        self.investigation_result = None

    def investigate(self) -> dict:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Sherlock config not found: {self.config_path}")
        self.result = AutoResearch.run_config(self.config_path)
        self.fit_result = self.result["fit"]
        self.investigation_result = self.result["investigation"]
        return self.result
