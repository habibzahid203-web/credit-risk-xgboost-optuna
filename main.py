"""
DAY 3 PROJECT — Credit Risk Classification Engine
Goal: Ek banking-style "credit default risk" predictor banana.
      1) Pehle standard Decision Tree (sklearn)
      2) Phir usay swap kar ke XGBoost (extreme gradient boosting)
      3) Optuna se Bayesian hyperparameter tuning (tree depth,
         learning rate, L1/L2 regularization)
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)  # optuna ka bohat sara extra output chup karo
np.random.seed(42)

# ============================================================
# STEP 1: Realistic Credit Risk Dataset banate hain
# ============================================================
# Har row ek loan applicant hai. Target: 'default' -> 1 = loan
# wapas nahi kar paya (risky), 0 = safely wapas kar diya.

n = 3000

income = np.random.normal(75000, 30000, n).clip(15000, 300000)
loan_amount = np.random.normal(25000, 15000, n).clip(1000, 150000)
credit_score = np.random.normal(650, 90, n).clip(300, 850)
debt_to_income = np.random.beta(2, 5, n) * 0.8          # 0 se 0.8 tak, real duniya jaisi distribution
previous_defaults = np.random.poisson(0.3, n).clip(0, 5)
employment_years = np.random.exponential(5, n).clip(0, 35)
age = np.random.normal(38, 12, n).clip(18, 75)

# Default hone ka "asal" (non-linear) risk formula -- isay hum
# jaan-boojh kar NON-LINEAR bana rahe hain (tree models isay
# achi tarah pakadte hain, linear regression nahi pakad pata)
risk_score = (
    -2.5 * (credit_score - 650) / 90
    + 2.8 * debt_to_income
    + 0.9 * previous_defaults
    - 1.2 * (income - 75000) / 30000
    + 0.6 * (loan_amount - 25000) / 15000
    - 0.4 * (employment_years - 5) / 5
    + 1.5 * np.where(age < 25, (25 - age) / 10, 0)          # bohat jawan applicants zyada risky (non-linear kink)
    + 1.5 * np.where(credit_score < 500, (500 - credit_score) / 100, 0)  # bohat kam credit score = extra risk (non-linear)
    + np.random.normal(0, 1.0, n)                            # random noise
)

# Probability mein convert karo (sigmoid), phir 0/1 default banao
# shift=4.0 se realistic ~19-20% default rate milti hai (real banking data jaisi)
probability_default = 1 / (1 + np.exp(-(risk_score - 4.0)))
default = (np.random.rand(n) < probability_default).astype(int)

df = pd.DataFrame({
    'income': income.round(0),
    'loan_amount': loan_amount.round(0),
    'credit_score': credit_score.round(0),
    'debt_to_income': debt_to_income.round(3),
    'previous_defaults': previous_defaults,
    'employment_years': employment_years.round(1),
    'age': age.round(0),
    'default': default
})

print("===== CREDIT RISK DATASET (pehle 5 rows) =====")
print(df.head())
print(f"\nTotal applicants: {len(df)}")
print(f"Default rate: {df['default'].mean()*100:.1f}% ({df['default'].sum()} out of {len(df)})")


# ============================================================
# STEP 2: Train/Test Split
# ============================================================
X = df.drop(columns=['default'])
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")


# ============================================================
# STEP 3: BASELINE — Standard Decision Tree (Scikit-Learn)
# ============================================================
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)
dt_probs = dt_model.predict_proba(X_test)[:, 1]

print("\n===== BASELINE: Decision Tree Performance =====")
print(f"Accuracy : {accuracy_score(y_test, dt_preds):.4f}")
print(f"F1 Score  : {f1_score(y_test, dt_preds):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, dt_probs):.4f}")


# ============================================================
# STEP 4: SWAP — XGBoost (Extreme Gradient Boosting), default settings
# ============================================================
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

print("\n===== XGBoost (default hyperparameters) Performance =====")
print(f"Accuracy : {accuracy_score(y_test, xgb_preds):.4f}")
print(f"F1 Score  : {f1_score(y_test, xgb_preds):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, xgb_probs):.4f}")


# ============================================================
# STEP 5: OPTUNA — Bayesian Hyperparameter Tuning
# ============================================================
# Optuna khud-ba-khud "best combination" dhoondta hai tree depth,
# learning rate, aur L1/L2 regularization ke liye -- bajaye is ke
# ke hum khud manually har combination try karein (jo bohat slow
# hota hai), Optuna "smart search" (Bayesian) karta hai.

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),   # L1 regularization
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True), # L2 regularization
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    return auc


print("\n===== OPTUNA: Bayesian Hyperparameter Search shuru ho raha hai... =====")
print("Optimization Metric: ROC-AUC (NOT raw accuracy -- chosen specifically because")
print("the dataset is imbalanced, and accuracy can be misleading in that case)")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40, show_progress_bar=False)

print(f"\nBest ROC-AUC found: {study.best_value:.4f}")
print("Best hyperparameters:")
for key, val in study.best_params.items():
    print(f"  {key}: {val}")


# ============================================================
# STEP 6: Best Hyperparameters se FINAL model train karo
# ============================================================
best_params = study.best_params
best_params['eval_metric'] = 'logloss'
best_params['random_state'] = 42

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train)

final_preds = final_model.predict(X_test)
final_probs = final_model.predict_proba(X_test)[:, 1]

print("\n===== FINAL TUNED XGBoost Performance =====")
print(f"Accuracy : {accuracy_score(y_test, final_preds):.4f}")
print(f"F1 Score  : {f1_score(y_test, final_preds):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, final_probs):.4f}")

print("\n===== Detailed Classification Report =====")
print(classification_report(y_test, final_preds, target_names=['No Default', 'Default']))


# ============================================================
# STEP 7: Teeno models ka COMPARISON (side by side)
# ============================================================
print("\n===== FINAL COMPARISON: Decision Tree vs XGBoost vs Tuned XGBoost =====")
comparison = pd.DataFrame({
    'Model': ['Decision Tree (baseline)', 'XGBoost (default)', 'XGBoost (Optuna-tuned)'],
    'Accuracy': [
        accuracy_score(y_test, dt_preds),
        accuracy_score(y_test, xgb_preds),
        accuracy_score(y_test, final_preds)
    ],
    'F1 Score': [
        f1_score(y_test, dt_preds),
        f1_score(y_test, xgb_preds),
        f1_score(y_test, final_preds)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, dt_probs),
        roc_auc_score(y_test, xgb_probs),
        roc_auc_score(y_test, final_probs)
    ]
})
print(comparison.to_string(index=False))


# ============================================================
# STEP 8: Feature Importance (kaunse features sabse zyada matter karte hain)
# ============================================================
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n===== FEATURE IMPORTANCE (Tuned XGBoost ke mutabiq) =====")
print(importance.to_string(index=False))


# ============================================================
# STEP 9: Custom applicant predict karne ka function
# ============================================================
def predict_credit_risk(income, loan_amount, credit_score, debt_to_income,
                         previous_defaults, employment_years, age):
    input_df = pd.DataFrame([{
        'income': income, 'loan_amount': loan_amount, 'credit_score': credit_score,
        'debt_to_income': debt_to_income, 'previous_defaults': previous_defaults,
        'employment_years': employment_years, 'age': age
    }])
    prob = final_model.predict_proba(input_df)[0, 1]
    prediction = "DEFAULT RISK (Loan reject/review karein)" if prob > 0.5 else "SAFE (Loan approve ho sakta hai)"
    return prob, prediction


print("\n===== CUSTOM APPLICANT PREDICTIONS =====")
test_applicants = [
    {"income": 90000, "loan_amount": 15000, "credit_score": 750, "debt_to_income": 0.15,
     "previous_defaults": 0, "employment_years": 8, "age": 35},
    {"income": 25000, "loan_amount": 40000, "credit_score": 480, "debt_to_income": 0.6,
     "previous_defaults": 2, "employment_years": 1, "age": 22},
]

for applicant in test_applicants:
    prob, verdict = predict_credit_risk(**applicant)
    print(f"\nApplicant: {applicant}")
    print(f"  Default Probability: {prob*100:.1f}%  ->  {verdict}")
