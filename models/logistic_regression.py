"""
models/logistic_regression.py
Logistic Regression classifier with hyperparameter config.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from features.feature_engineering import build_preprocessing_pipeline


def build_model() -> Pipeline:
    """Returns a full preprocessing + LR pipeline."""
    return Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )),
    ])
