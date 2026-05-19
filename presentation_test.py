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
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Matplotlib not found. Run:  pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

# import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not found. Run:  pip install xgboost")
    XGBOOST_AVAILABLE = False

# 1.  LOAD DATA
DATA_PATH = "hp_contention_dataset.csv"
OUTPUT_DIR = Path("outputs")

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


def save_comparison_table(comparison_df, output_dir):
    """Save the split comparison as CSV and as a matplotlib table image."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "model_comparison_by_split.csv"
    png_path = output_dir / "model_comparison_by_split.png"

    comparison_df.to_csv(csv_path)
    print(f"\nSaved comparison CSV: {csv_path}")

    if not MATPLOTLIB_AVAILABLE:
        print("Skipped table image export because matplotlib is unavailable.")
        return

    display_df = comparison_df.copy().map(lambda x: f"{x:.4f}")
    row_labels = [[model, split] for model, split in display_df.index]
    cell_text = [[model, split, *values] for (model, split), values in zip(row_labels, display_df.values)]
    col_labels = ["Model", "Split", *display_df.columns.tolist()]

    fig_height = max(2.6, 0.85 + 0.5 * len(display_df))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dbeafe")
            cell.set_text_props(weight="bold")
        if col == -1:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold")

    ax.set_title("Model Comparison by Split", fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison image: {png_path}")


def save_feature_importance_chart(importances, output_dir):
    """Save the Random Forest feature importances as a bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not MATPLOTLIB_AVAILABLE:
        print("Skipped feature importance image export because matplotlib is unavailable.")
        return

    sorted_importances = importances.sort_values(ascending=True)
    fig_height = max(3.6, 0.55 * len(sorted_importances) + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.barh(sorted_importances.index, sorted_importances.values, color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importances", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for i, value in enumerate(sorted_importances.values):
        ax.text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    png_path = output_dir / "random_forest_feature_importance.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved feature importance image: {png_path}")


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
save_feature_importance_chart(importances, OUTPUT_DIR)

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
print("FINAL COMPARISON by model and split")
print("=" * 65)

all_split_results = []
for results in [lr_results, rf_results, xgb_results]:
    if results:
        for result in results:
            model_name, split_label = result["model"].rsplit(" (", 1)
            split_label = split_label.rstrip(")")
            all_split_results.append({
                "model": model_name,
                "split": split_label,
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
            })

if all_split_results:
    comparison = pd.DataFrame(all_split_results)
    comparison["split"] = pd.Categorical(
        comparison["split"],
        categories=["70/30 split", "80/20 split"],
        ordered=True,
    )
    comparison = comparison.sort_values(["model", "split"]).set_index(["model", "split"])
    print(comparison.map(lambda x: f"{x:.4f}").to_string())
    save_comparison_table(comparison, OUTPUT_DIR)
