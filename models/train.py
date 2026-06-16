"""
models/train.py
Trains all three classifiers, saves them to disk, and prints a summary.
"""

import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.generate_dataset import generate_dataset
from features.feature_engineering import prepare_data
from evaluation.metrics import evaluate_model

import models.logistic_regression as lr_mod
import models.decision_tree as dt_mod
import models.random_forest as rf_mod

MODELS_DIR = Path(__file__).parent
DATA_PATH  = Path(__file__).parent.parent / "data" / "sample_dataset.csv"


def load_or_generate_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        print(f"Loading dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    print("Generating synthetic dataset …")
    df = generate_dataset(n_samples=10_000)
    df.to_csv(DATA_PATH, index=False)
    return df


def train_all():
    df = load_or_generate_data()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}\n")

    registry = {
        "random_forest":       rf_mod.build_model(),
        "logistic_regression": lr_mod.build_model(),
        "decision_tree":       dt_mod.build_model(),
    }

    results = {}
    for name, pipeline in registry.items():
        t0 = time.time()
        # These pipelines already include their own preprocessing, so fit on raw X
        # However, prepare_data already engineers features — we strip preprocessing here
        # to avoid double-scaling. For simplicity, refit on raw df.
        pass

    # Re-train using raw features so each pipeline handles its own preprocessing
    from features.feature_engineering import RAW_FEATURES, TARGET
    X_raw = df[RAW_FEATURES]
    y_all = df[TARGET]
    X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
        X_raw, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    for name, pipeline in registry.items():
        print(f"Training {name} …", end=" ", flush=True)
        t0 = time.time()
        pipeline.fit(X_raw_train, y_raw_train)
        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s")

        metrics = evaluate_model(pipeline, X_raw_test, y_raw_test, name)
        results[name] = metrics

        save_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, save_path)
        print(f"  Saved → {save_path}")

    # Summary table
    print("\n" + "="*65)
    print(f"{'Model':<25} {'Acc':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-"*65)
    for name, m in results.items():
        print(
            f"{name:<25} "
            f"{m['accuracy']:>6.1%} "
            f"{m['precision']:>7.1%} "
            f"{m['recall']:>7.1%} "
            f"{m['f1']:>7.1%} "
            f"{m['roc_auc']:>7.3f}"
        )
    print("="*65)
    return results


if __name__ == "__main__":
    train_all()
