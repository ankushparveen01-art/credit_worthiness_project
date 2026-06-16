"""
models/decision_tree.py
Decision Tree classifier with depth control to avoid overfitting.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from features.feature_engineering import build_preprocessing_pipeline


def build_model() -> Pipeline:
    return Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            min_samples_split=40,
            class_weight="balanced",
            criterion="gini",
            random_state=42,
        )),
    ])
