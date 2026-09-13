import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

class DriftAnalyzer:
    def __init__(self, threshold=.05):
        self.threshold = threshold

    def compare(self, reference, production):
        out=[]
        for col in reference.columns:
            if col not in production.columns: continue
            a,b=reference[col].dropna(),production[col].dropna()
            if len(a)==0 or len(b)==0: continue
            if pd.api.types.is_numeric_dtype(reference[col]):
                stat,p=ks_2samp(a,b); test="ks_2samp"
            else:
                cats=sorted(set(a.unique())|set(b.unique()))
                table=np.array([[sum(a==c) for c in cats],[sum(b==c) for c in cats]])
                if table.shape[1]<2: continue
                stat,p,_,_=chi2_contingency(table); test="chi2"
            out.append({"feature":col,"test":test,"statistic":float(stat),
                        "p_value":float(p),"drift":bool(p<self.threshold)})
        return out

    def diagnose(self, baseline, production, drift):
        degraded=[]
        for m,b in baseline.items():
            p=production.get(m)
            if b is None or p is None: continue
            bad = (m in {"rmse","mae","mape"} and p>b) or (m=="r2" and p<b)
            if bad:
                degraded.append({"metric":m,"baseline":b,"production":p,
                                 "change_pct":((p-b)/abs(b)*100) if b else None})
        drifted=[x for x in drift if x["drift"]]
        return {"status":"degraded" if degraded else "healthy",
                "performance_degradation":degraded,
                "drifted_features":drifted,
                "summary":f"{len(degraded)} metrics degraded; {len(drifted)} features drifted."}
