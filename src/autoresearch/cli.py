import argparse
import logging
import os
import warnings

# Hide MLflow's optional agent hint before importing the package that initializes MLflow.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from .core.research import AutoResearch

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message="`artifact_path` is deprecated.*")
    warnings.filterwarnings("ignore", message="Saving scikit-learn models in the pickle.*")
    warnings.filterwarnings("ignore", message="Encountered an unexpected error while inferring pip requirements.*")
    warnings.filterwarnings("ignore", message="The specified dataset source can be interpreted in multiple ways.*")
    warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column.*")
    p=argparse.ArgumentParser(prog="ml-sherlock")
    s=p.add_subparsers(dest="command",required=True)
    f=s.add_parser("fit"); f.add_argument("--train",required=True); f.add_argument("--target",required=True)
    i=s.add_parser("investigate"); i.add_argument("--train",required=True); i.add_argument("--production",required=True); i.add_argument("--target",required=True)
    run=s.add_parser("run", help="Run a complete investigation from a YAML configuration file")
    run.add_argument("--config", required=True)
    for command in (f, i):
        command.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                             help="MLflow tracking backend URI (default: sqlite:///mlflow.db)")
    a=p.parse_args()
    if a.command == "run":
        from ml_sherlock import Sherlock

        result = Sherlock(config=a.config).investigate()
        research = result["investigation"]["research"]
        print(f"\nCompleted {research['iterations_completed']} experiments. {research['recommendation']}")
        print(f"Report: {result['investigation']['report']}")
        return
    r=AutoResearch(a.target, tracking_uri=a.tracking_uri)
    if a.command=="fit": print(r.fit(a.train))
    else: r.fit(a.train); print(r.investigate(a.train,a.production))
