"""
Contention Risk Prediction - GPU Clusters
CSC 428 Machine Learning Final Project
Pragesh Adhikari & Prasant Koirala

Runs 3 algorithms:
  1. Logistic Regression  (simple baseline)
  2. Random Forest        (mid-level)
  3. XGBoost              (advanced)

Each is evaluated with:
  - 70/30 train/test split
  - 80/20 train/test split
  - 10-fold cross validation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

# import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not found. Run:  pip install xgboost")
    XGBOOST_AVAILABLE = False

# 1.  LOAD DATA
DATA_PATH = "hp_contention_dataset.csv"

print("=" * 65)
print("LOADING DATA")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nContention label distribution:")
print(df["contention_label"].value_counts())
print(f"\nClass balance: {df['contention_label'].mean()*100:.1f}% positive (contention)")

# 2.  PREPARE FEATURES
print("\n" + "=" * 65)
print("PREPARING FEATURES")
print("=" * 65)

# Encode gpu_model (text) → numbers
le = LabelEncoder()
df["gpu_model_encoded"] = le.fit_transform(df["gpu_model"])
print(f"GPU model encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Select features — only scheduling-observable ones
FEATURES = [
    "gpu_model_encoded",
    "cpu_request",
    "gpu_request",
    "worker_num",
    "concurrent_spot_jobs",
    "concurrent_spot_workers",
    "spot_load_ratio",
    "arrival_rate_1h",
]

TARGET = "contention_label"

# Drop rows with missing values in selected features
before = len(df)
df = df.dropna(subset=FEATURES + [TARGET])
after = len(df)
print(f"\nDropped {before - after} rows with missing values ({after:,} rows remain)")

X = df[FEATURES]
y = df[TARGET]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Features used: {FEATURES}")

# 3.  HELPER — print evaluation metrics cleanly
def evaluate(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1}


def run_splits_and_cv(name, model, X, y):
    """Run 70/30, 80/20, and 10-fold CV for a given model."""
    results = []
    print(f"\n{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")

    for test_size, label in [(0.30, "70/30 split"), (0.20, "80/20 split")]:
        # NOTE: shuffle=False preserves temporal order (important for traces)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n  [{label}]  train={len(X_train):,}  test={len(X_test):,}")
        r = evaluate(name + f" ({label})", y_test, y_pred)
        results.append(r)

    # 10-Fold Cross Validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"\n  [10-Fold CV]")
    print(f"  F1 per fold: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  Mean F1    : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results


# 4.  ALGORITHM 1 — LOGISTIC REGRESSION
print("\n\n" + "=" * 65)
print("ALGORITHM 1: LOGISTIC REGRESSION")
print("=" * 65)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_results = run_splits_and_cv("Logistic Regression", lr_model, X, y)

# 5.  ALGORITHM 2 — RANDOM FOREST
print("\n\n" + "=" * 65)
print("ALGORITHM 2: RANDOM FOREST")
print("=" * 65)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_results = run_splits_and_cv("Random Forest", rf_model, X, y)

# Feature importance
X_train_full, _, y_train_full, _ = train_test_split(X, y, test_size=0.20, shuffle=False)
rf_model.fit(X_train_full, y_train_full)
importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
print("\n  Feature Importances (higher = more predictive):")
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "█" * int(imp * 40)
    print(f"    {feat:<30} {imp:.4f}  {bar}")

# 6.  ALGORITHM 3 — XGBOOST
print("\n\n" + "=" * 65)
print("ALGORITHM 3: XGBOOST (Gradient Boosting)")
print("=" * 65)

xgb_results = []
if XGBOOST_AVAILABLE:
    # scale_pos_weight handles class imbalance automatically
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos if pos > 0 else 1
    print(f"  scale_pos_weight = {spw:.2f}  (compensates for class imbalance)")

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb_results = run_splits_and_cv("XGBoost", xgb_model, X, y)
else:
    print("  Skipped — install xgboost first.")

# 7.  FINAL COMPARISON TABLE
print("\n\n" + "=" * 65)
print("FINAL COMPARISON with 80/20 split")
print("=" * 65)

all_80_20 = []
for results in [lr_results, rf_results, xgb_results]:
    if results:
        # grab the 80/20 result (index 1)
        all_80_20.append(results[1])

if all_80_20:
    comparison = pd.DataFrame(all_80_20).set_index("model")
    comparison = comparison.map(lambda x: f"{x:.4f}")
    print(comparison.to_string())

