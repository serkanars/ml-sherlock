# Real-data examples

These examples turn public datasets into the same two-file contract:

- `data/train.csv`: reference data used to select the initial model.
- `data/production.csv`: a later period or a different domain used for drift investigation.

Install the project and the Parquet reader used by the NYC Taxi example:

```bash
pip install -e ".[examples]"
```

Run any scenario from the repository root:

```bash
python examples/nyc_taxi/prepare.py
ml-sherlock run --config examples/nyc_taxi/sherlock.yaml
```

| Scenario | Reference -> production | Target | Drift type |
|---|---|---|---|
| [NYC Taxi](nyc_taxi/README.md) | January 2024 -> January 2025 | Trip duration | Temporal, demand, policy |
| [Citi Bike](citi_bike/README.md) | January 2024 -> January 2025 | Ride duration | Temporal, rider and station mix |
| [Seoul Bike](seoul_bike/README.md) | Dec 2017-Aug 2018 -> Sep-Nov 2018 | Hourly rentals | Seasonal and weather |
| [California Housing](california_housing/README.md) | Northern -> southern California | Median house value | Geographic domain shift |

The preparation scripts use fixed seeds and deterministic filters. Downloaded
raw files, generated CSVs, reports, models, and MLflow databases are ignored by
Git. LLM planning is configured independently in each scenario's YAML; disable
it for strictly deterministic runs or enable it when testing an LLM planner.

Each scenario writes to its own `examples/<scenario>/mlflow.db`. Point the UI at
that database rather than the repository-root database, for example:

```bash
mlflow ui --backend-store-uri sqlite:///examples/nyc_taxi/mlflow.db --port 5001
```
