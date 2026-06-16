"""
main.py
Full end-to-end pipeline:
  1. Generate (or load) dataset
  2. Feature engineering
  3. Train all three classifiers
  4. Evaluate with Precision, Recall, F1, ROC-AUC
  5. Save trained models and plots
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.generate_dataset import generate_dataset
from features.feature_engineering import RAW_FEATURES, TARGET
from models.train import load_or_generate_data
import models.logistic_regression as lr_mod
import models.decision_tree as dt_mod
import models.random_forest as rf_mod
from evaluation.metrics import evaluate_model, compare_models
from utils.visualize import (
    plot_roc_curves,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall,
    plot_score_distribution,
)
from features.feature_engineering import ENGINEERED_FEATURES

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).parent / "models"


def run_pipeline():
    print("\n" + "█"*55)
    print("  CREDIT WORTHINESS PREDICTION — FULL PIPELINE")
    print("█"*55)

    # 1. Data
    print("\n[1/4] Loading dataset …")
    df = load_or_generate_data()
    print(f"      {len(df):,} records  |  creditworthy: {df[TARGET].mean():.1%}")

    X_raw = df[RAW_FEATURES]
    y     = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Train
    print("\n[2/4] Training models …")
    registry = {
        "random_forest":       rf_mod.build_model(),
        "logistic_regression": lr_mod.build_model(),
        "decision_tree":       dt_mod.build_model(),
    }
    for name, pipeline in registry.items():
        print(f"      → {name}")
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib")

    # 3. Evaluate
    print("\n[3/4] Evaluating …")
    results = {}
    for name, pipeline in registry.items():
        results[name] = evaluate_model(pipeline, X_test, y_test, name)

    compare_models(results)

    # 4. Plots
    print("\n[4/4] Generating plots …")
    plot_score_distribution(df)
    plot_roc_curves(results)
    plot_precision_recall(results)
    for name, m in results.items():
        plot_confusion_matrix(m["confusion_matrix"], name)
    plot_feature_importance(registry["random_forest"], ENGINEERED_FEATURES)

    print("\n✓ Pipeline complete. Run  python app.py  to launch the dashboard.\n")


if __name__ == "__main__":
    run_pipeline()
