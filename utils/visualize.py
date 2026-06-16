"""
utils/visualize.py
Plotting helpers: ROC curves, confusion matrix, feature importance, distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

PLOTS_DIR = Path(__file__).parent.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

PALETTE = {
    "random_forest":       "#3266ad",
    "logistic_regression": "#1D9E75",
    "decision_tree":       "#D85A30",
    "positive":            "#1D9E75",
    "negative":            "#E24B4A",
}


def _save(fig, filename: str):
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot → {path}")
    plt.close(fig)


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(results: dict, save: bool = True):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")
    for name, m in results.items():
        color = PALETTE.get(name, "#888")
        ax.plot(
            m["fpr"], m["tpr"],
            label=f"{name}  (AUC = {m['roc_auc']:.3f})",
            color=color, linewidth=2,
        )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Classifiers")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    sns.despine(fig)
    if save:
        _save(fig, "roc_curves.png")
    return fig


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, model_name: str, save: bool = True):
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = np.array([
        [f"TN\n{cm[0,0]:,}", f"FP\n{cm[0,1]:,}"],
        [f"FN\n{cm[1,0]:,}", f"TP\n{cm[1,1]:,}"],
    ])
    cmap_data = np.array([[cm[0,0], cm[0,1]], [cm[1,0], cm[1,1]]])
    sns.heatmap(
        cmap_data, annot=labels, fmt="", cmap="Blues",
        xticklabels=["Pred: No", "Pred: Yes"],
        yticklabels=["Act: No",  "Act: Yes"],
        ax=ax, linewidths=0.5, cbar=False,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    if save:
        _save(fig, f"cm_{model_name.replace(' ','_').lower()}.png")
    return fig


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(pipeline, feature_names: list, save: bool = True):
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "feature_importances_"):
        print("  Model does not expose feature_importances_. Skipping.")
        return None

    importances = clf.feature_importances_
    n = min(len(feature_names), len(importances))
    names = feature_names[:n]
    imps  = importances[:n]
    idx = np.argsort(imps)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [names[i] for i in idx],
        imps[idx],
        color=PALETTE["random_forest"], edgecolor="none",
    )
    ax.set_xlabel("Gini importance")
    ax.set_title("Feature Importance — Random Forest")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    sns.despine(fig)
    if save:
        _save(fig, "feature_importance.png")
    return fig


# ── Precision–Recall curve ────────────────────────────────────────────────────

def plot_precision_recall(results: dict, save: bool = True):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, m in results.items():
        color = PALETTE.get(name, "#888")
        ax.plot(m["rec_curve"], m["prec_curve"], label=name, color=color, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curves")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    sns.despine(fig)
    if save:
        _save(fig, "precision_recall.png")
    return fig


# ── Credit score distribution ─────────────────────────────────────────────────

def plot_score_distribution(df, save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color, name in [(0, PALETTE["negative"], "Not creditworthy"),
                                (1, PALETTE["positive"], "Creditworthy")]:
        subset = df[df["creditworthy"] == label]["credit_score"]
        ax.hist(subset, bins=30, alpha=0.65, color=color, label=name, edgecolor="none")
    ax.set_xlabel("Credit Score")
    ax.set_ylabel("Count")
    ax.set_title("Credit Score Distribution by Class")
    ax.legend()
    sns.despine(fig)
    if save:
        _save(fig, "score_distribution.png")
    return fig
