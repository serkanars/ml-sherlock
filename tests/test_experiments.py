import pandas as pd

from ml_sherlock.evidence import Evidence
from ml_sherlock.investigation.experiments import ExperimentRunner
from ml_sherlock.models.trainer import BaselineTrainer


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


def _evidence_driven_fixture():
    reference = pd.DataFrame({
        "x": range(80),
        "noise": [value % 7 for value in range(80)],
        "group": ["a", "b"] * 40,
        "target": [value * 2 + value % 3 for value in range(80)],
    })
    production = pd.DataFrame({
        "x": range(80, 160),
        "noise": [value % 7 for value in range(80, 160)],
        "group": ["a", "b"] * 40,
        "target": [value * 2 + value % 5 for value in range(80, 160)],
    })
    trainer = BaselineTrainer(random_state=11, candidates=["random_forest"])
    baseline = trainer.fit_full(reference, "target", "random_forest")
    return reference, production, trainer, baseline


def test_recent_window_retraining_uses_only_recent_adaptation_rows():
    reference, production, trainer, baseline = _evidence_driven_fixture()
    runner = ExperimentRunner(trainer, random_state=7)

    result = runner.run_action(
        "recent_window_retraining", reference, production, "target", baseline, [],
        hypothesis_id="target_relationship_shift", evidence_ids=["target-1"],
    )

    assert result["dataset_rows_used"]["reference_training"] == 0
    assert result["dataset_rows_used"]["total_training"] == result["adaptation_rows"]
    assert result["hypothesis_id"] == "target_relationship_shift"
    assert result["evidence_ids"] == ["target-1"]


def test_segment_retraining_uses_rows_selected_by_segment_evidence():
    reference, production, trainer, baseline = _evidence_driven_fixture()
    evidence = Evidence(
        id="segment-b", type="segment_degradation", metric="rmse_degradation_pct",
        value=80.0, feature="group", segment="group=b", severity="high",
        metadata={"definition": {"category": "b"}},
    )
    runner = ExperimentRunner(trainer, random_state=7)

    result = runner.run_action(
        "segment_retraining", reference, production, "target", baseline, [],
        hypothesis_id="segment_specific_degradation",
        evidence_ids=[evidence.id], evidence=[evidence],
    )

    rows = result["dataset_rows_used"]
    assert result["segment"] == "group=b"
    assert 0 < rows["production_adaptation"] < rows["production_adaptation_available"]
    assert rows["total_training"] == len(reference) + rows["production_adaptation"]


def test_feature_subset_search_is_bounded_reproducible_and_traceable():
    reference, production, trainer, baseline = _evidence_driven_fixture()
    evidence = Evidence(
        id="noise-error", type="feature_error_relationship", metric="association",
        value=.8, feature="noise", severity="high",
    )
    runner = ExperimentRunner(trainer, random_state=7)
    kwargs = {
        "hypothesis_id": "covariate_shift",
        "evidence_ids": [evidence.id],
        "evidence": [evidence],
    }

    first = runner.run_action(
        "feature_subset_search", reference, production, "target", baseline, [], **kwargs
    )
    second = runner.run_action(
        "feature_subset_search", reference, production, "target", baseline, [], **kwargs
    )

    assert first["searched_feature_subsets"] == 2
    assert first["candidate_metrics"] == second["candidate_metrics"]
    assert first["random_seed"] == 11
    assert all(set(candidate["features"]) <= {"x", "noise", "group"}
               for candidate in first["candidates"])


def test_every_new_action_records_the_common_experiment_contract():
    reference, production, trainer, baseline = _evidence_driven_fixture()
    runner = ExperimentRunner(trainer, random_state=7)
    evidence = Evidence(
        id="segment-a", type="segment_degradation", metric="rmse_degradation_pct",
        value=50.0, feature="group", segment="group=a",
        metadata={"definition": {"category": "a"}},
    )
    required = {
        "hypothesis_id", "evidence_ids", "action", "dataset_rows_used",
        "used_features", "model", "evaluation_metrics", "improvement_pct",
        "accepted", "random_seed",
    }
    for action in (
        "segment_retraining", "recent_window_retraining", "feature_subset_search"
    ):
        result = runner.run_action(
            action, reference, production, "target", baseline, ["noise"],
            hypothesis_id="hypothesis-1", evidence_ids=[evidence.id],
            evidence=[evidence],
        )
        assert required <= set(result)
        assert result["status"] == ("validated" if result["accepted"] else "rejected")


def test_unregistered_action_cannot_be_executed():
    reference, production, trainer, baseline = _evidence_driven_fixture()
    runner = ExperimentRunner(trainer)

    try:
        runner.run_action(
            "execute_python", reference, production, "target", baseline, []
        )
    except ValueError as exc:
        assert "Unsupported experiment action" in str(exc)
    else:
        raise AssertionError("unregistered action was executed")
