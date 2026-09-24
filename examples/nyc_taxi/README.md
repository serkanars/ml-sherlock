# NYC Taxi

This scenario downloads official NYC TLC Green Taxi records and predicts trip
duration. January 2024 is the reference period and January 2025 is the observed
production period. The one-year separation captures changes in trip mix,
locations, demand, and the operating environment without adding synthetic noise.

Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

```bash
python examples/nyc_taxi/prepare.py
ml-sherlock run --config examples/nyc_taxi/sherlock.yaml
```

Open this example's isolated MLflow store with:

```bash
mlflow ui --backend-store-uri sqlite:///examples/nyc_taxi/mlflow.db --port 5001
```

The preparation step removes invalid dates, trips shorter than one minute or
longer than two hours, implausible distances, and invalid passenger counts. Fare
components are intentionally excluded because they are observed after the trip.
