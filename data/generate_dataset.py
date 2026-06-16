"""
data/generate_dataset.py
Generates a synthetic credit dataset with realistic feature distributions.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_dataset(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Base features ---
    income = rng.lognormal(mean=13.5, sigma=0.6, size=n_samples).clip(200_000, 5_000_000)
    credit_score = rng.normal(650, 90, size=n_samples).clip(300, 900).astype(int)
    on_time_pct = rng.beta(a=8, b=2, size=n_samples) * 100          # 0-100
    utilization = rng.beta(a=2, b=4, size=n_samples) * 100          # 0-100
    history_years = rng.gamma(shape=3, scale=3, size=n_samples).clip(0, 30)
    delinquencies = rng.negative_binomial(n=1, p=0.7, size=n_samples).clip(0, 10)
    total_debt = income * rng.beta(a=2, b=5, size=n_samples) * 0.8

    # --- Label generation: weighted logistic rule ---
    log_odds = (
        (credit_score - 650) / 120 * 2.5
        + (on_time_pct - 80) / 20 * 1.8
        - (utilization - 40) / 30 * 1.2
        - (total_debt / income - 0.3) / 0.3 * 1.5
        + (history_years - 7) / 7 * 0.8
        - delinquencies * 0.9
    )
    prob = 1 / (1 + np.exp(-log_odds))
    creditworthy = (rng.random(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "annual_income":       income.round(2),
        "total_debt":          total_debt.round(2),
        "credit_score":        credit_score,
        "on_time_pct":         on_time_pct.round(2),
        "credit_utilization":  utilization.round(2),
        "history_years":       history_years.round(1),
        "delinquencies":       delinquencies,
        "creditworthy":        creditworthy,
    })
    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "sample_dataset.csv"
    df = generate_dataset(n_samples=10_000)
    df.to_csv(out_path, index=False)
    print(f"Dataset saved → {out_path}  |  shape: {df.shape}")
    print(df["creditworthy"].value_counts(normalize=True).rename("proportion").to_string())
