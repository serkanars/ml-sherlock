from ml_sherlock import Sherlock

# Edit sherlock.example.yaml (or copy it to sherlock.yaml) instead of changing
# Python code for normal runs.
sherlock = Sherlock(config="sherlock.example.yaml")
result = sherlock.investigate()
print(result)
