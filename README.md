# Credit Risk Classification Engine

A loan default risk classifier that progresses from a simple, interpretable baseline to a fully-tuned gradient boosting model — the same kind of pipeline used by banks and lending platforms to automate approve/reject decisions.

## What This Project Does

- Generates a realistic synthetic credit risk dataset (income, credit score, debt-to-income ratio, previous defaults, etc.) with a non-linear, realistic default probability model
- Trains a baseline **Decision Tree** (Scikit-Learn) for interpretability
- Upgrades to **XGBoost** (gradient boosting) for improved performance
- Runs automated **Bayesian hyperparameter optimization** using Optuna (40 trials), tuning tree depth, learning rate, and L1/L2 regularization (`reg_alpha`, `reg_lambda`)
- Optimizes explicitly for **ROC-AUC** rather than raw accuracy, since credit risk datasets are typically imbalanced (few defaults relative to non-defaults) — accuracy alone can be a misleading metric in this setting
- Includes a custom prediction function to assess the risk of a new, user-defined loan applicant

## Why ROC-AUC Instead of Accuracy?

In imbalanced classification (e.g., 80% non-default / 20% default), a model that always predicts "safe" can appear highly accurate while catching zero actual defaults. ROC-AUC measures how well the model ranks risky vs. safe applicants across all thresholds, making it a far more meaningful metric for this kind of problem.

## Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Decision Tree (baseline) | 84.3% | 0.569 | 0.840 |
| XGBoost (default) | 87.2% | 0.632 | 0.892 |
| XGBoost (Optuna-tuned) | 87.3% | 0.635 | **0.897** |

## Feature Importance

Credit score was the single strongest predictor of default risk, accounting for ~39% of the tuned model's decision-making — consistent with real-world credit scoring practices.

## Tech Stack

- Python
- Scikit-Learn (Decision Tree, evaluation metrics)
- XGBoost (gradient boosting)
- Optuna (Bayesian hyperparameter optimization)

## Setup

```bash
pip install numpy pandas scikit-learn xgboost optuna
python day3_credit_risk_xgboost.py
```

---

*Part of a self-driven 60-day AI/ML Engineering learning journey, focused on implementing and understanding tree-based ensemble methods from first principles before relying on high-level APIs.*
