"""
models/random_forest.py
Random Forest classifier — primary model with feature importance support.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from features.feature_engineering import build_preprocessing_pipeline


def build_model() -> Pipeline:
    return Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])
