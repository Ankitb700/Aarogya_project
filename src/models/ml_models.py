from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _xgb_regressor():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42)
    except Exception:
        return RandomForestRegressor(n_estimators=200, random_state=42)


def _xgb_classifier():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42)
    except Exception:
        return RandomForestClassifier(n_estimators=200, random_state=42)


def feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = {"subject_id", "trial_id", target, "cv_load_class"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def train_grouped_model(df: pd.DataFrame, target: str, task: str = "regression") -> tuple[Pipeline, dict]:
    data = df.dropna(subset=[target]).copy()
    cols = feature_columns(data, target)
    X = data[cols]
    y = data[target]
    groups = data["subject_id"] if "subject_id" in data else np.arange(len(data))
    if data["subject_id"].nunique() > 1 and len(data) >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        train_idx = np.arange(len(data))
        test_idx = np.arange(len(data))
    estimator = _xgb_classifier() if task == "classification" else _xgb_regressor()
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    if task == "classification":
        metrics = {"accuracy": float(accuracy_score(y.iloc[test_idx], pred))}
    else:
        metrics = {
            "mae": float(mean_absolute_error(y.iloc[test_idx], pred)),
            "rmse": float(np.sqrt(mean_squared_error(y.iloc[test_idx], pred))),
            "r2": float(r2_score(y.iloc[test_idx], pred)) if len(test_idx) > 1 else np.nan,
        }
    metrics["features"] = cols
    metrics["n_rows"] = int(len(data))
    return model, metrics


def save_model(model: Pipeline, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
