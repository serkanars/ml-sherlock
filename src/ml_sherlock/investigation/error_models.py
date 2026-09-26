"""Diagnostic model for predicting the primary model's absolute error."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ..evidence import Evidence, make_evidence_id
from .feature_errors import _absolute_error


class ErrorModelAnalyzer:
    """Fit a holdout-validated model that predicts absolute primary-model error."""

    def __init__(
        self,
        *,
        validation_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int | None = 8,
        min_samples_leaf: int = 5,
        top_features: int = 10,
        min_feature_importance: float = 0.1,
        min_validation_r2: float = 0.1,
    ):
        if not 0 < validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1")
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if top_features < 1:
            raise ValueError("top_features must be at least 1")
        if not 0 <= min_feature_importance <= 1:
            raise ValueError("min_feature_importance must be between 0 and 1")
        self.validation_size = validation_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.top_features = top_features
        self.min_feature_importance = min_feature_importance
        self.min_validation_r2 = min_validation_r2

    def analyze(self, features: pd.DataFrame, y_true, primary_predictions) -> dict:
        """Train and evaluate a diagnostic error model on a deterministic holdout."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if features.empty or not len(features.columns):
            raise ValueError("features must contain rows and at least one column")

        error_target = _absolute_error(y_true, primary_predictions, len(features))
        valid = np.isfinite(error_target.to_numpy(dtype=float))
        clean_features = features.reset_index(drop=True).loc[valid].reset_index(drop=True)
        clean_target = error_target.loc[valid].reset_index(drop=True)
        if len(clean_features) < 5:
            raise ValueError("at least five valid observations are required")
        if clean_target.nunique() < 2:
            raise ValueError("error target must contain at least two distinct values")

        train_x, validation_x, train_y, validation_y = train_test_split(
            clean_features,
            clean_target,
            test_size=self.validation_size,
            random_state=self.random_state,
        )
        numeric = clean_features.select_dtypes(include=["number"]).columns.tolist()
        categorical = [column for column in clean_features.columns if column not in numeric]
        preprocessor = _preprocessor(numeric, categorical)
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        model.fit(train_x, train_y)

        validation_predictions = model.predict(validation_x)
        metrics = _validation_metrics(validation_y, validation_predictions)
        importance = _aggregate_feature_importance(
            model.named_steps["preprocessor"],
            model.named_steps["model"].feature_importances_,
            numeric,
            categorical,
        )
        for rank, item in enumerate(importance, start=1):
            item["rank"] = rank
        top = importance[: self.top_features]

        reliable = metrics["r2"] >= self.min_validation_r2
        evidence = [
            self._evidence(item, metrics, len(clean_features), len(train_x), len(validation_x))
            for item in top
            if reliable and item["importance"] >= self.min_feature_importance
        ]
        return {
            "model_type": "random_forest_regressor",
            "diagnostic_only": True,
            "target": "absolute_error",
            "split": {
                "strategy": "holdout",
                "random_state": self.random_state,
                "train_size": int(len(train_x)),
                "validation_size": int(len(validation_x)),
            },
            "validation_metrics": metrics,
            "feature_importance": importance,
            "top_error_driving_features": top,
            "evidence": evidence,
            "interpretation": (
                "Predictive associations with absolute primary-model error; "
                "these results do not establish causal explanations."
            ),
        }

    def _evidence(self, item, metrics, sample_size, train_size, validation_size):
        metadata = {
            "rank": item["rank"],
            "feature_importance": item["importance"],
            "error_model": "random_forest_regressor",
            "error_target": "absolute_error",
            "validation_metrics": metrics,
            "holdout": {
                "train_size": train_size,
                "validation_size": validation_size,
                "random_state": self.random_state,
            },
            "interpretation": (
                "The feature has predictive association with absolute model error on a "
                "holdout-validated diagnostic model; this is not a causal explanation."
            ),
        }
        return Evidence(
            id=make_evidence_id(
                "feature_error_relationship",
                "error_model_feature_importance",
                feature=item["feature"],
            ),
            type="feature_error_relationship",
            metric="error_model_feature_importance",
            value=item["importance"],
            feature=item["feature"],
            threshold=self.min_feature_importance,
            severity=_importance_severity(item["importance"]),
            sample_size_production=sample_size,
            metadata=metadata,
        )


def _preprocessor(numeric, categorical):
    transformers = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers)


def _validation_metrics(y_true, predictions):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }


def _aggregate_feature_importance(preprocessor, transformed_importance, numeric, categorical):
    aggregated = []
    if numeric:
        numeric_slice = preprocessor.output_indices_["numeric"]
        for feature, importance in zip(numeric, transformed_importance[numeric_slice]):
            aggregated.append({"feature": str(feature), "importance": float(importance)})

    if categorical:
        categorical_slice = preprocessor.output_indices_["categorical"]
        categorical_importance = transformed_importance[categorical_slice]
        onehot = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        offset = 0
        for feature, categories in zip(categorical, onehot.categories_):
            width = len(categories)
            importance = float(np.sum(categorical_importance[offset : offset + width]))
            aggregated.append({"feature": str(feature), "importance": importance})
            offset += width

    aggregated.sort(key=lambda item: (-item["importance"], item["feature"]))
    return aggregated


def _importance_severity(importance):
    if importance < 0.2:
        return "low"
    if importance < 0.4:
        return "medium"
    if importance < 0.7:
        return "high"
    return "critical"
