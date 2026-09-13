import pandas as pd
from autoresearch.monitoring.drift import DriftAnalyzer

def test_no_drift():
    r=DriftAnalyzer().compare(pd.DataFrame({"x":[1,2,3,4,5]}),pd.DataFrame({"x":[1,2,3,4,5]}))
    assert len(r)==1 and r[0]["drift"] is False
