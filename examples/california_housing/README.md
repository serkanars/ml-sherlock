# California Housing

This scenario uses `sklearn.datasets.fetch_california_housing`. Census blocks
north of latitude 36 form the reference domain; blocks south of latitude 36
form production. This is a geographic transfer test, not a temporal backtest.

Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

```bash
python examples/california_housing/prepare.py
ml-sherlock run --config examples/california_housing/sherlock.yaml
```
