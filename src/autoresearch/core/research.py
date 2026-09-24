from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from ..data.profiler import DataProfiler
from ..data.tracking import DatasetTracker
from ..models.trainer import BaselineTrainer
from ..monitoring.drift import DriftAnalyzer
from ..reporting.report import ReportBuilder
from ..investigation import ExperimentRunner, ResearchEngine, ResearchLoop
from ..config import InvestigationConfig
from ..llm import LLMConfig, create_provider

class AutoResearch:
    def __init__(self, target, metric="rmse", experiment_name="ml-sherlock",
                 tracking_uri="sqlite:///mlflow.db", random_state=42,
                 drift_p_value_threshold=.05, candidates=None,
                 adaptation_fraction=.5, min_improvement_pct=1.0, llm: LLMConfig | None = None,
                 max_experiments=5, allowed_actions=None):
        self.target, self.metric = target, metric.lower()
        self.profiler = DataProfiler()
        self.tracker = DatasetTracker(tracking_uri, experiment_name)
        self.trainer = BaselineTrainer(random_state, candidates, self.metric)
        self.drift = DriftAnalyzer(drift_p_value_threshold)
        self.reporter = ReportBuilder()
        self.research_engine = ResearchEngine(planner=create_provider(llm), allowed_actions=allowed_actions)
        self.experiments = ExperimentRunner(
            self.trainer, self.metric, random_state, adaptation_fraction, min_improvement_pct
        )
        self.loop = ResearchLoop(self.research_engine, self.experiments, max_experiments,
                                 allowed_actions, random_state)
        self.model = None
        self.baseline_metrics = None
        self.baseline_candidates = []
        self.baseline_run_id = None

    def fit(self, train_path):
        train = pd.read_csv(train_path)
        if self.target not in train:
            raise ValueError(f"Target '{self.target}' not found.")
        profile = self.profiler.profile(train, self.target)
        dataset = self.tracker.log_dataset(train, Path(train_path).stem,
                                            str(Path(train_path).resolve()), "training")
        result = self.trainer.fit(train, self.target)
        self.model, self.baseline_metrics = result.model, result.metrics
        self.baseline_candidates = result.candidates or []
        run_id = self.tracker.log_run(dataset, result.metrics, result.params,
                                      result.model, profile, self.baseline_candidates)
        self.baseline_run_id = run_id
        return {"run_id": run_id, "metrics": result.metrics, "dataset": dataset,
                "candidates": self.baseline_candidates}

    @classmethod
    def from_config(cls, config_path):
        config = InvestigationConfig.from_yaml(config_path)
        return cls(config.target, experiment_name=config.experiment_name,
                   tracking_uri=config.tracking_uri, random_state=config.random_state,
                   drift_p_value_threshold=config.drift_p_value_threshold,
                   candidates=list(config.model_candidates), metric=config.selection_metric,
                   adaptation_fraction=config.adaptation_fraction,
                   min_improvement_pct=config.min_improvement_pct, llm=config.llm,
                   max_experiments=config.max_experiments,
                   allowed_actions=list(config.allowed_actions)), config

    @classmethod
    def run_config(cls, config_path):
        research, config = cls.from_config(config_path)
        fit_result = research.fit(config.train_path)
        investigation = research.investigate(
            config.train_path, config.production_path, config.report_path
        )
        return {"fit": fit_result, "investigation": investigation}

    def investigate(self, reference_path, production_path,
                    report_path="autoresearch_report.html"):
        if self.model is None:
            raise RuntimeError("Call fit() before investigate().")
        ref, prod = pd.read_csv(reference_path), pd.read_csv(production_path)
        prod, final_evaluation = train_test_split(prod, test_size=.2, random_state=self.experiments.random_state)
        production_metrics = self.trainer.evaluate(self.model, prod, self.target)
        drift = self.drift.compare(ref.drop(columns=[self.target]),
                                   prod.drop(columns=[self.target]))
        diagnosis = self.drift.diagnose(self.baseline_metrics, production_metrics, drift)
        self.loop.iteration_logger = lambda result, evidence: self.tracker.log_research_iteration(
            self.baseline_run_id, result, evidence
        )
        research = self.loop.run(ref, prod, self.target, self.model, diagnosis, drift, final_evaluation)
        self.recommended_model = research.pop("_champion")
        research["initial_candidates"] = self.baseline_candidates
        decision = research["decision"]
        decision["target"] = self.target
        decision["reference_profile"] = self.profiler.profile(ref, self.target)
        decision["production_profile"] = self.profiler.profile(prod, self.target)
        decision["training_data"] = {
            "reference_path": str(Path(reference_path).resolve()),
            "production_path": str(Path(production_path).resolve()),
            "reference_rows": len(ref), "development_rows": len(prod),
            "final_evaluation_fraction": .2,
            "adaptation_fraction": self.experiments.adaptation_fraction,
            "split_seed": self.experiments.random_state,
            "policy": "Reference plus fixed production adaptation rows, each included once; refit the accepted model family and feature set.",
        }
        model_path = Path(report_path).with_suffix(".joblib")
        decision_path = Path(report_path).with_suffix(".json")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.recommended_model, model_path)
        decision["model_path"] = str(model_path.resolve())
        decision_path.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
        # sklearn models are logged to MLflow; do not expose in the public result.
        for experiment in research["experiments"]:
            experiment.pop("_model", None)
        report = self.reporter.build(report_path, self.target,
                                     self.baseline_metrics, production_metrics,
                                     drift, diagnosis, research)
        self.tracker.log_final_report(self.baseline_run_id, decision, report, model_path, decision_path)
        return {"production_metrics": production_metrics, "drift": drift,
                "diagnosis": diagnosis, "research": research, "report": report}
