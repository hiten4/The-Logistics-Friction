
# Run this AFTER running all 4 individual model files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, roc_curve,
    precision_recall_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH = "final_dataset.csv"

df = pd.read_csv(DATA_PATH)

TARGET = "is_delayed"

from settings import DATA_PATH, TARGET, FEATURES

state_cols    = [c for c in df.columns if c.startswith("customer_state_") or c.startswith("seller_state_")]
category_cols = [c for c in df.columns if c.startswith("category_group_")]
ALL_FEATURES  = FEATURES + state_cols + category_cols

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()
bool_cols = X.select_dtypes(bool).columns
X[bool_cols] = X[bool_cols].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

print("Training all models...")

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])

rf = RandomForestClassifier(
    n_estimators=200, max_depth=12, min_samples_leaf=20,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=neg/pos, random_state=42,
    eval_metric="logloss", verbosity=0
)

lgbm = LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    class_weight="balanced", random_state=42, verbosity=-1
)

models = {
    "Logistic Regression": lr,
    "Random Forest":       rf,
    "XGBoost":             xgb,
    "LightGBM":            lgbm,
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    results[name] = {
        "prob":    y_prob,
        "pred":    y_pred,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc":  average_precision_score(y_test, y_prob),
    }
    print(f"  {name} done")

print("\n" + "="*55)
print("  MODEL COMPARISON SUMMARY — CHECKPOINT 2")
print("="*55)
print(f"  {'Model':<25} {'ROC-AUC':>9} {'PR-AUC':>9}")
print(f"  {'─'*45}")
for name, r in results.items():
    print(f"  {name:<25} {r['roc_auc']:>9.4f} {r['pr_auc']:>9.4f}")

colors = ["#7c6af7", "#4ade80", "#fb923c", "#f472b6"]
plt.figure(figsize=(8, 6))
for (name, r), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r["prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})", color=color, lw=2)
plt.plot([0,1],[0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — All Models (Checkpoint 2)")
plt.legend()
plt.tight_layout()
plt.savefig("c2_roc_comparison.png")
plt.show()

plt.figure(figsize=(8, 6))
for (name, r), color in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, r["prob"])
    plt.plot(rec, prec, label=f"{name} (PR-AUC={r['pr_auc']:.3f})", color=color, lw=2)
plt.axhline(y_test.mean(), ls="--", color="gray", label=f"Baseline ({y_test.mean():.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve — All Models (Checkpoint 2)")
plt.legend()
plt.tight_layout()
plt.savefig("c2_pr_comparison.png")
plt.show()

# Find best model by PR-AUC
best_name = max(results, key=lambda n: results[n]["pr_auc"])
best_prob  = results[best_name]["prob"]
print(f"\nBest model: {best_name}")
print("Tuning decision threshold to maximize Recall on Delayed class...\n")

thresholds = np.arange(0.1, 0.6, 0.05)
print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'─'*44}")

best_thresh = 0.5
best_recall = 0

for t in thresholds:
    preds = (best_prob >= t).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds)
    f = f1_score(y_test, preds, zero_division=0)
    print(f"  {t:>10.2f} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    if r > best_recall:
        best_recall = r
        best_thresh = t

print(f"\nRecommended threshold: {best_thresh:.2f}  (maximizes Recall on Delayed class)")
print("Lower threshold = catch more delays but more false alarms")
print("Higher threshold = fewer false alarms but miss more delays")
print("\nDone! All plots saved to Downloads.")
