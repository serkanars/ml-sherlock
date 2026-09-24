from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


@dataclass
class TrainingResult:
    model: object
    metrics: dict
    params: dict
    candidates: list | None = None


class BaselineTrainer:
    """Trains comparable, reproducible sklearn model candidates."""

    SUPPORTED_MODELS = ("random_forest", "extra_trees", "xgboost", "lightgbm")

    def __init__(self, random_state=42, candidates=None, selection_metric="rmse"):
        self.random_state = random_state
        self.candidates = candidates or list(self.SUPPORTED_MODELS)
        self.selection_metric = selection_metric
        invalid = set(self.candidates) - set(self.SUPPORTED_MODELS)
        if invalid:
            raise ValueError(f"Unsupported model candidates: {sorted(invalid)}")

    def _estimator(self, name):
        common = {"n_estimators": 300, "random_state": self.random_state, "n_jobs": -1}
        if name == "random_forest":
            return RandomForestRegressor(**common)
        if name == "extra_trees":
            return ExtraTreesRegressor(**common)
        if name == "xgboost":
            return XGBRegressor(
                **common,
                objective="reg:squarederror",
                tree_method="hist",
                verbosity=0,
            )
        if name == "lightgbm":
            return LGBMRegressor(**common, verbosity=-1)
        raise ValueError(f"Unsupported model candidate: {name}")

    def _pipeline(self, X, candidate):
        numeric = X.select_dtypes(include=["number"]).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        transformers = []
        if numeric:
            transformers.append(("num", SimpleImputer(strategy="median"), numeric))
        if categorical:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical))
        return Pipeline([
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("model", self._estimator(candidate)),
        ])

    def fit(self, df, target):
        train, validation = train_test_split(df, test_size=.2, random_state=self.random_state)
        results = self.fit_and_evaluate(train, validation, target, self.candidates)
        selected = self.select_best(results)
        selected.candidates = [
            {
                "model": result.params["model"],
                "metrics": result.metrics,
                "params": result.params,
                "selected": result is selected,
            }
            for result in results
        ]
        # Validation chooses the candidate; deployment trains that candidate on
        # all reference rows, not only the selection split.
        selected.model = self.fit_full(df, target, selected.params["model"])
        return selected

    def fit_and_evaluate(self, train_df, evaluation_df, target, candidates=None):
        results = []
        for candidate in candidates or self.candidates:
            model = self.fit_full(train_df, target, candidate)
            results.append(TrainingResult(
                model=model,
                metrics=self.evaluate(model, evaluation_df, target),
                params={"model": candidate, "n_estimators": 300, "random_state": self.random_state},
            ))
        return results

    def fit_full(self, df, target, candidate):
        X, y = df.drop(columns=[target]), df[target]
        model = self._pipeline(X, candidate)
        model.fit(X, y)
        return model

    @staticmethod
    def model_name(model):
        estimator = model.named_steps["model"]
        if isinstance(estimator, ExtraTreesRegressor):
            return "extra_trees"
        if isinstance(estimator, RandomForestRegressor):
            return "random_forest"
        if isinstance(estimator, XGBRegressor):
            return "xgboost"
        if isinstance(estimator, LGBMRegressor):
            return "lightgbm"
        raise ValueError("Unsupported baseline estimator.")

    def select_best(self, results):
        if not results:
            raise ValueError("At least one candidate result is required.")
        def score(result):
            value = result.metrics[self.selection_metric]
            if value is None:
                return float("-inf") if self.selection_metric == "r2" else float("inf")
            return value
        return sorted(
            results,
            key=score,
            reverse=self.selection_metric == "r2",
        )[0]

    def evaluate(self, model, df, target):
        return self._metrics(df[target], model.predict(df.drop(columns=[target])))

    @staticmethod
    def feature_importance(model, limit=30):
        """Return human-readable transformed-feature importances when available."""
        estimator = model.named_steps["model"]
        if not hasattr(estimator, "feature_importances_"):
            return []
        names = model.named_steps["preprocessor"].get_feature_names_out()
        pairs = sorted(zip(names, estimator.feature_importances_), key=lambda item: item[1], reverse=True)
        return [{"feature": str(name), "importance": float(score)} for name, score in pairs[:limit]]

    @staticmethod
    def _metrics(y, p):
        y, p = np.asarray(y), np.asarray(p)
        nz = y != 0
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "mae": float(mean_absolute_error(y, p)),
            "r2": float(r2_score(y, p)),
            "mape": float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100) if nz.any() else None,
        }
