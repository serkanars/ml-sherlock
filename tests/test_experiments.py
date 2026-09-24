import pandas as pd

from autoresearch.investigation.experiments import ExperimentRunner
from autoresearch.models.trainer import BaselineTrainer


def test_retraining_experiment_uses_unseen_holdout():
    reference = pd.DataFrame({"x": range(20), "target": [value * 2 for value in range(20)]})
    production = pd.DataFrame({"x": range(20, 40), "target": [value * 2 for value in range(20, 40)]})
    trainer = BaselineTrainer(random_state=1, candidates=["random_forest"])
    baseline = trainer.fit(reference, "target").model
    result = ExperimentRunner(trainer, random_state=1).validate_retraining(
        reference, production, "target", baseline
    )
    assert result["adaptation_rows"] + result["holdout_rows"] == len(production)
    assert result["candidates"][0]["model"] == "random_forest"


def test_supported_models_fit_mixed_features_and_report_their_names():
    frame = pd.DataFrame({
        "numeric": range(60),
        "category": ["a", "b", "c"] * 20,
        "target": [value * 1.5 + value % 3 for value in range(60)],
    })
    for candidate in BaselineTrainer.SUPPORTED_MODELS:
        trainer = BaselineTrainer(random_state=7, candidates=[candidate])
        result = trainer.fit(frame, "target")
        assert trainer.model_name(result.model) == candidate
        assert result.params["model"] == candidate
        assert set(result.metrics) == {"rmse", "mae", "r2", "mape"}
        assert all(value is None or pd.notna(value) for value in result.metrics.values())


def test_baseline_fit_preserves_every_candidate_result():
    frame = pd.DataFrame({
        "x": range(50),
        "target": [value * 2 + value % 4 for value in range(50)],
    })
    candidates = ["random_forest", "xgboost", "lightgbm"]
    result = BaselineTrainer(random_state=7, candidates=candidates).fit(frame, "target")

    assert [item["model"] for item in result.candidates] == candidates
    assert sum(item["selected"] for item in result.candidates) == 1
    assert all(set(item["metrics"]) == {"rmse", "mae", "r2", "mape"}
               for item in result.candidates)
