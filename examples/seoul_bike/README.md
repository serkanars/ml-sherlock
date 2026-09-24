# Seoul Bike Sharing Demand

This UCI regression dataset contains hourly rental counts with weather, season,
holiday, and service-status features. December 2017 through August 2018 forms
the reference set; September through November 2018 forms production. The split
creates a real seasonal shift without modifying feature values.

Source: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

```bash
python examples/seoul_bike/prepare.py
ml-sherlock run --config examples/seoul_bike/sherlock.yaml
```
