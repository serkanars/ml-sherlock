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
