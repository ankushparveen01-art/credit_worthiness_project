"""
features/feature_engineering.py
Transforms raw financial columns into model-ready features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived features on top of the raw financial columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Debt-to-income ratio
        df["dti_ratio"] = (df["total_debt"] / df["annual_income"]).round(4)

        # 2. Payment score: reward long, consistent payment history
        df["payment_score"] = (df["on_time_pct"] / 100) * df["history_years"]

        # 3. Utilization bucket (ordinal: 0=low, 1=medium, 2=high, 3=over)
        df["util_bucket"] = pd.cut(
            df["credit_utilization"],
            bins=[-1, 30, 60, 80, 101],
            labels=[0, 1, 2, 3],
        ).astype(int)

        # 4. Delinquency flag
        df["delinq_flag"] = (df["delinquencies"] > 0).astype(int)

        # 5. Log-transformed income (compress right tail)
        df["log_income"] = np.log1p(df["annual_income"])

        # 6. Risk composite: combines three negative signals
        util_norm = df["credit_utilization"] / 100
        df["risk_composite"] = df["dti_ratio"] * (df["delinquencies"] + 1) * util_norm

        return df

    def get_feature_names_out(self, input_features=None):
        derived = [
            "dti_ratio", "payment_score", "util_bucket",
            "delinq_flag", "log_income", "risk_composite",
        ]
        base = list(input_features) if input_features is not None else []
        return base + derived


# Columns used by models after engineering
RAW_FEATURES = [
    "annual_income", "total_debt", "credit_score",
    "on_time_pct", "credit_utilization", "history_years", "delinquencies",
]

ENGINEERED_FEATURES = [
    "credit_score", "on_time_pct", "dti_ratio", "credit_utilization",
    "delinquencies", "history_years", "total_debt", "risk_composite",
    "log_income", "payment_score", "util_bucket", "delinq_flag",
]

TARGET = "creditworthy"


def build_preprocessing_pipeline() -> Pipeline:
    """Returns a sklearn Pipeline: feature engineering → standard scaling."""
    return Pipeline([
        ("engineer", CreditFeatureEngineer()),
        ("scaler",   StandardScaler()),
    ])


def prepare_data(df: pd.DataFrame):
    """Split df into feature matrix X and label vector y."""
    engineer = CreditFeatureEngineer()
    df_eng = engineer.transform(df[RAW_FEATURES])
    X = df_eng[ENGINEERED_FEATURES]
    y = df[TARGET]
    return X, y
