# Citi Bike

This scenario uses official Jersey City-prefixed Citi Bike trip-history files,
which are compact enough for a local example while retaining the production
schema. It predicts ride duration using January 2024 as reference and January
2025 as production.

Source: https://citibikenyc.com/system-data

```bash
python examples/citi_bike/prepare.py
ml-sherlock run --config examples/citi_bike/sherlock.yaml
```

Staff and test-station rides are already removed by the publisher. The script
also removes incomplete records and durations outside 1-120 minutes.
