import pandas as pd

class DataProfiler:
    def profile(self, df, target=None):
        features = []
        for col in df.columns:
            s = df[col]
            item = {"name": col, "dtype": str(s.dtype),
                    "missing": int(s.isna().sum()),
                    "missing_rate": float(s.isna().mean()),
                    "unique": int(s.nunique(dropna=True))}
            if pd.api.types.is_numeric_dtype(s) and s.notna().any():
                item.update(mean=float(s.mean()), std=float(s.std()),
                            min=float(s.min()), max=float(s.max()))
            features.append(item)
        return {"rows": len(df), "columns": len(df.columns),
                "target": target, "features": features}
