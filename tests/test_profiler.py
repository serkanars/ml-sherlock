import pandas as pd
from ml_sherlock.data.profiler import DataProfiler

def test_profile():
    r=DataProfiler().profile(pd.DataFrame({"x":[1,2,None],"target":[10,20,30]}),"target")
    assert r["rows"]==3 and r["features"][0]["missing"]==1
