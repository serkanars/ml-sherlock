# ML-Sherlock 🕵️

Investigate why machine learning models fail in production.

MVP:
CSV → profiling → MLflow dataset/run → baseline model → production comparison → drift detection → diagnosis → HTML report.

MLflow uses a local SQLite database (`sqlite:///mlflow.db`) by default. To use a
different backend, pass `--tracking-uri` to the CLI or `tracking_uri` to
`AutoResearch`.

Install:
`pip install -e .`

Run (copy and edit `sherlock.example.yaml` for your data):
`ml-sherlock run --config sherlock.example.yaml`

The YAML file contains the dataset paths, target, tracking backend, model seed,
model candidates, drift threshold, and report location. Supported regression
candidates are `random_forest`, `extra_trees`, `xgboost`, and `lightgbm`.
The report includes research hypotheses and the next recommended experiment,
each linked to measured evidence.

Optional LLM planning is configured under `llm` in the YAML file. Ollama requires
no additional Python package. For OpenAI or an OpenAI-compatible endpoint, install
`pip install -e ".[openai]"` and set the API-key environment variable named by
`llm.api_key_env`; never store the key in YAML.

Set `llm.enabled: true` to activate Ollama. The CLI then logs provider setup,
each planning request, selected action, and any fallback to deterministic planning.

```python
from autoresearch import AutoResearch, LLMConfig

research = AutoResearch(
    target="y", metric="rmse", tracking_uri="http://mlflow-server:5000",
    llm=LLMConfig(provider="ollama", model="qwen3:8b", base_url="http://localhost:11434"),
)
```

Experiment actions use the same production adaptation/holdout split for comparison:
`retrain_recent_data` retrains the baseline model family with reference plus adaptation
data; `drop_drifted_features` uses that family after removing drifted inputs;
`model_search` compares all configured model candidates on the adapted data.
Trials compare against the last accepted model. Accepted model families and feature
removals carry forward; rejected trials leave that state unchanged. The tree models
are refitted on reference plus adaptation data, with each row included once (this is
not incremental tree training). Repeating an action varies the training seed.
Twenty percent of production rows are reserved before diagnosis and planning for
independent final evaluation. The remaining rows are split into adaptation and
selection holdout. Only the final evaluation determines the deployment recommendation.

MLflow records `holdout_baseline_*`, the actual `holdout_candidate_*` metrics even
when a trial is rejected, and `holdout_recommended_*` for the deployment recommendation.
The logged model and `selected_model` identify the tested candidate;
`recommended_model` identifies the current accepted model family. In cumulative
runs, `holdout_baseline_*` refers to the previous accepted model; `parent_iteration`
records that lineage. All candidate scores are stored in `research/candidates.json`.
Existing runs are not rewritten.

The final HTML report includes independent evaluation, exact estimator parameters,
required features, observed data shifts and monitoring recommendations. Companion
`.joblib` and `.json` files contain the selected fitted pipeline and decision record;
all three are also logged under a separate `research-final-report` MLflow run's
`decision` artifacts, tagged with the original run ID.
The saved candidate is not automatically deployed: check `deployment_status`.
