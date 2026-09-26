"""Public ML-Sherlock API."""

import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from .sherlock import Sherlock
from .config import SherlockConfig

__all__ = ["Sherlock", "SherlockConfig"]
