"""
evaluation/metrics.py
Computes and prints all classification metrics; returns a results dict.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def evaluate_model(pipeline, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a fitted sklearn pipeline on test data.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc,
                    confusion_matrix, fpr, tpr, thresholds,
                    prec_curve, rec_curve, pr_thresholds
    """
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_proba)
    cm    = confusion_matrix(y_test, y_pred)

    fpr, tpr, roc_thresh   = roc_curve(y_test, y_proba)
    pr_prec, pr_rec, pr_th = precision_recall_curve(y_test, y_proba)

    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion matrix:\n{cm}")
    print(f"\n  Classification report:\n{classification_report(y_test, y_pred)}")

    return {
        "accuracy":       acc,
        "precision":      prec,
        "recall":         rec,
        "f1":             f1,
        "roc_auc":        auc,
        "confusion_matrix": cm,
        "fpr":            fpr,
        "tpr":            tpr,
        "roc_thresholds": roc_thresh,
        "prec_curve":     pr_prec,
        "rec_curve":      pr_rec,
        "pr_thresholds":  pr_th,
    }


def compare_models(results: dict) -> None:
    """Print a side-by-side comparison table for multiple evaluated models."""
    header = f"{'Model':<25} {'Acc':>7} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}"
    print("\n" + "="*67)
    print(header)
    print("-"*67)
    for name, m in results.items():
        print(
            f"{name:<25} "
            f"{m['accuracy']:>7.2%} "
            f"{m['precision']:>8.2%} "
            f"{m['recall']:>8.2%} "
            f"{m['f1']:>8.2%} "
            f"{m['roc_auc']:>8.4f}"
        )
    print("="*67)
