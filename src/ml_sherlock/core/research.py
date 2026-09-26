from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from ..data.profiler import DataProfiler
from ..data.tracking import DatasetTracker
from ..models.trainer import BaselineTrainer
from ..monitoring.drift import DriftAnalyzer, PredictionDriftAnalyzer, TargetDriftAnalyzer
from ..reporting.report import ReportBuilder
from ..investigation import (
    DiagnosisEngine,
    ExperimentRunner,
    ResearchEngine,
    ResearchLoop,
    SegmentAnalyzer,
)
from ..config import SherlockConfig
from ..llm import LLMConfig, create_provider
from ..evidence import Evidence, EvidenceStore, make_evidence_id

class ResearchRunner:
    def __init__(self, target, metric="rmse", experiment_name="ml-sherlock",
                 tracking_uri="sqlite:///artifacts/mlflow.db", random_state=42,
                 drift_p_value_threshold=.05, candidates=None,
                 adaptation_fraction=.5, min_improvement_pct=1.0, llm: LLMConfig | None = None,
                 max_experiments=5, allowed_actions=None, drift_enabled=True,
                 error_analysis_enabled=True, error_metrics=None, segment_config=None,
                 drift_multiple_testing="benjamini_hochberg"):
        self.target, self.metric = target, metric.lower()
        self.profiler = DataProfiler()
        self.tracker = DatasetTracker(tracking_uri, experiment_name)
        self.trainer = BaselineTrainer(random_state, candidates, self.metric)
        self.drift = DriftAnalyzer(
            alpha=drift_p_value_threshold, multiple_testing=drift_multiple_testing
        )
        self.target_drift = TargetDriftAnalyzer(
            alpha=drift_p_value_threshold, multiple_testing=drift_multiple_testing
        )
        self.prediction_drift = PredictionDriftAnalyzer(
            alpha=drift_p_value_threshold, multiple_testing=drift_multiple_testing
        )
        self.drift_enabled = drift_enabled
        self.error_analysis_enabled = error_analysis_enabled
        self.error_metrics = error_metrics or ["rmse", "mae", "mape", "r2"]
        self.segment_config = segment_config
        self.segment_analyzer = (
            SegmentAnalyzer.from_config(segment_config)
            if segment_config is not None and segment_config.enabled
            else None
        )
        self.diagnosis_engine = DiagnosisEngine()
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
        config = SherlockConfig.from_yaml(config_path)
        return cls.from_typed_config(config), config

    @classmethod
    def from_typed_config(cls, config: SherlockConfig):
        return cls(
            config.data.target,
            experiment_name=config.tracking.experiment,
            tracking_uri=config.tracking.uri,
            random_state=config.models.random_state,
            drift_p_value_threshold=config.investigation.drift.p_value_threshold,
            drift_multiple_testing=config.investigation.drift.multiple_testing,
            candidates=list(config.models.candidates),
            metric=config.models.selection_metric,
            adaptation_fraction=config.experiments.adaptation_fraction,
            min_improvement_pct=config.experiments.min_improvement_pct,
            llm=config.llm if config.llm.enabled else None,
            max_experiments=config.experiments.max_iterations,
            allowed_actions=list(config.experiments.allowed_actions),
            drift_enabled=config.investigation.drift.enabled,
            error_analysis_enabled=config.investigation.error_analysis.enabled,
            error_metrics=list(config.investigation.error_analysis.metrics),
            segment_config=config.investigation.segments,
        )

    @classmethod
    def run_config(cls, config_path):
        if isinstance(config_path, SherlockConfig):
            config = config_path
            research = cls.from_typed_config(config)
        else:
            research, config = cls.from_config(config_path)
        fit_result = research.fit(config.data.train)
        investigation = research.investigate(
            config.data.train, config.data.production, config.report.output
        )
        return {"fit": fit_result, "investigation": investigation}

    def investigate(self, reference_path, production_path,
                    report_path="artifacts/report.html"):
        if self.model is None:
            raise RuntimeError("Call fit() before investigate().")
        ref, prod = pd.read_csv(reference_path), pd.read_csv(production_path)
        prod, final_evaluation = train_test_split(prod, test_size=.2, random_state=self.experiments.random_state)
        production_metrics = self.trainer.evaluate(self.model, prod, self.target)
        target_drift = self.target_drift.analyze(ref[self.target], prod[self.target], self.target)
        prediction_drift = self.prediction_drift.analyze(
            self.model,
            ref.drop(columns=[self.target]),
            prod.drop(columns=[self.target]),
        )
        segment_analysis = None
        if self.segment_analyzer is not None:
            reference_features = ref.drop(columns=[self.target])
            production_features = prod.drop(columns=[self.target])
            segment_analysis = self.segment_analyzer.analyze(
                reference_features,
                production_features,
                ref[self.target],
                self.model.predict(reference_features),
                prod[self.target],
                self.model.predict(production_features),
                metric=self.metric,
            )
        drift = self.drift.compare(ref.drop(columns=[self.target]),
                                   prod.drop(columns=[self.target])) if self.drift_enabled else []
        evidence_store = EvidenceStore([target_drift, prediction_drift])
        evidence_store.extend(
            _feature_drift_evidence(
                drift,
                ref.drop(columns=[self.target]),
                prod.drop(columns=[self.target]),
                self.drift.effect_threshold,
            )
        )
        if segment_analysis is not None:
            evidence_store.extend(segment_analysis["evidence"])
        baseline_for_diagnosis = {
            key: value for key, value in self.baseline_metrics.items()
            if self.error_analysis_enabled and key in self.error_metrics
        }
        diagnosis_engine = getattr(self, "diagnosis_engine", None) or DiagnosisEngine()
        diagnosis = diagnosis_engine.summarize(
            evidence_store, baseline_for_diagnosis, production_metrics
        )
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
        target_evidence = target_drift.to_dict()
        prediction_evidence = prediction_drift.to_dict()
        serialized_segment_analysis = None
        segment_evidence = []
        if segment_analysis is not None:
            segment_evidence = [item.to_dict() for item in segment_analysis["evidence"]]
            serialized_segment_analysis = {
                **segment_analysis,
                "evidence": segment_evidence,
            }
        return {"production_metrics": production_metrics, "drift": drift,
                "target_drift": target_evidence, "prediction_drift": prediction_evidence,
                "segment_analysis": serialized_segment_analysis,
                "evidence": evidence_store.to_dict(),
                "diagnosis": diagnosis, "research": research, "report": report}


def _feature_drift_evidence(results, reference, production, threshold):
    evidence = []
    for result in results:
        feature = result["feature"]
        evidence.append(
            Evidence(
                id=make_evidence_id(
                    "feature_drift", "distribution_effect_size", feature=feature
                ),
                type="feature_drift",
                metric="distribution_effect_size",
                value=result.get("effect_size"),
                feature=feature,
                threshold=threshold,
                severity=result["severity"],
                sample_size_reference=int(reference[feature].notna().sum()),
                sample_size_production=int(production[feature].notna().sum()),
                metadata=result,
            )
        )
    return evidence
