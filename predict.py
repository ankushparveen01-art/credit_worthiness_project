"""
predict.py
CLI tool to predict creditworthiness for a single applicant.

Usage:
    python predict.py --income 1800000 --debt 600000 --credit-score 720 \
                      --on-time-pct 92 --utilization 28 --history-years 8 \
                      --delinquencies 0 --dti 33 --model random_forest
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features.feature_engineering import RAW_FEATURES

MODELS_DIR = Path(__file__).parent / "models"
AVAILABLE   = ["random_forest", "logistic_regression", "decision_tree"]


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        print(f"[ERROR] Model file not found: {path}")
        print("  Run  python main.py  first to train and save models.")
        sys.exit(1)
    return joblib.load(path)


def predict(args):
    model = load_model(args.model)

    applicant = pd.DataFrame([{
        "annual_income":      args.income,
        "total_debt":         args.debt,
        "credit_score":       args.credit_score,
        "on_time_pct":        args.on_time_pct,
        "credit_utilization": args.utilization,
        "history_years":      args.history_years,
        "delinquencies":      args.delinquencies,
    }])

    prob  = model.predict_proba(applicant)[0, 1]
    label = model.predict(applicant)[0]

    verdict = "CREDITWORTHY ✓" if label == 1 else "NOT CREDITWORTHY ✗"
    bar     = "█" * int(prob * 40) + "░" * (40 - int(prob * 40))

    print("\n" + "="*55)
    print(f"  Model      : {args.model}")
    print(f"  Verdict    : {verdict}")
    print(f"  Probability: {prob:.1%}")
    print(f"  [{bar}]")
    print("="*55)

    # Simple factor breakdown
    factors = {
        "Credit score":       (args.credit_score, 700, True),
        "On-time payments":   (args.on_time_pct,  80,  True),
        "Credit utilization": (args.utilization,  30,  False),
        "Delinquencies":      (args.delinquencies, 0,  False),
    }
    print("\n  Key factors:")
    for fname, (val, threshold, higher_is_good) in factors.items():
        if higher_is_good:
            sign = "+" if val >= threshold else "-"
        else:
            sign = "+" if val <= threshold else "-"
        print(f"    [{sign}] {fname}: {val}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Credit Worthiness Predictor")
    p.add_argument("--income",        type=float, required=True,  help="Annual income (₹)")
    p.add_argument("--debt",          type=float, required=True,  help="Total debt (₹)")
    p.add_argument("--credit-score",  type=int,   required=True,  help="Credit score (300-900)")
    p.add_argument("--on-time-pct",   type=float, required=True,  help="On-time payment % (0-100)")
    p.add_argument("--utilization",   type=float, required=True,  help="Credit utilization % (0-100)")
    p.add_argument("--history-years", type=float, required=True,  help="Credit history in years")
    p.add_argument("--delinquencies", type=int,   required=True,  help="Number of delinquencies")
    p.add_argument("--model", choices=AVAILABLE, default="random_forest")
    return p.parse_args()


if __name__ == "__main__":
    predict(parse_args())
