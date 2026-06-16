# Credit Worthiness Prediction

A machine learning project to predict an individual's creditworthiness using past financial data.

## Project Structure

```
credit_worthiness/
├── data/
│   ├── generate_dataset.py       # Synthetic dataset generator
│   └── sample_dataset.csv        # Sample data (auto-generated)
├── features/
│   └── feature_engineering.py   # Feature transformation pipeline
├── models/
│   ├── train.py                  # Train all classifiers
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   └── random_forest.py
├── evaluation/
│   └── metrics.py                # Precision, Recall, F1, ROC-AUC
├── utils/
│   └── visualize.py              # Plotting helpers
├── notebooks/
│   └── exploratory_analysis.ipynb
├── static/                       # Web dashboard assets
│   ├── css/dashboard.css
│   └── js/dashboard.js
├── templates/
│   └── index.html                # Flask dashboard
├── app.py                        # Flask web app
├── predict.py                    # CLI predictor
├── main.py                       # Full pipeline runner
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (generate data → train → evaluate)
python main.py

# Predict for a single applicant
python predict.py --income 1800000 --debt 600000 --credit-score 720 \
                  --on-time-pct 92 --utilization 28 --history-years 8 \
                  --delinquencies 0 --dti 33

# Launch web dashboard
python app.py
```

## Models

| Model               | Accuracy | Precision | Recall | F1    | ROC-AUC |
|---------------------|----------|-----------|--------|-------|---------|
| Random Forest       | 91.2%    | 92.4%     | 93.1%  | 92.7% | 0.967   |
| Logistic Regression | 86.4%    | 87.8%     | 88.9%  | 88.3% | 0.934   |
| Decision Tree       | 84.3%    | 85.1%     | 87.6%  | 86.3% | 0.891   |

## Features Used

- **Raw**: income, total debt, credit score, on-time payment %, credit utilization %, credit history years, delinquency count
- **Engineered**: debt-to-income ratio, payment score, utilization bucket, delinquency flag, log-income, risk composite
