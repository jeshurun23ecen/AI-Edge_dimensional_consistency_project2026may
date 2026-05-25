

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

OUTPUT_DIR   = "results"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
os.makedirs(OUTPUT_DIR, exist_ok=True)

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


section("STEP 1 — Load Data")

df = pd.read_csv("C:/Users/Jeshurun Solomon/Desktop/Xgboost L&t/Piece_Dimension.csv")
print(f"  Shape   : {df.shape}")
print(f"  Columns : {df.columns.tolist()}")
print(f"\n  First 5 rows:\n{df.head()}")
print(f"\n  Missing values:\n{df.isnull().sum()}")


section("STEP 2 — Create Target Label (Consistent=1 / Inconsistent=0)")

dims   = ["Length", "Width", "Height"]
bounds = {}

for col in dims:
    mu    = df[col].mean()
    sigma = df[col].std()
    lo    = mu - 2 * sigma
    hi    = mu + 2 * sigma
    bounds[col] = (lo, hi)
    print(f"  {col:8s}: mean={mu:.3f}  std={sigma:.3f}  "
          f"→ acceptable range [{lo:.3f}, {hi:.3f}] mm")

df["passed"] = df.apply(
    lambda row: int(all(bounds[c][0] <= row[c] <= bounds[c][1] for c in dims)),
    axis=1
)

vc = df["passed"].value_counts()
print(f"\n  Consistent   (1): {vc.get(1, 0)} parts  ({vc.get(1,0)/len(df)*100:.1f}%)")
print(f"  Inconsistent (0): {vc.get(0, 0)} parts  ({vc.get(0,0)/len(df)*100:.1f}%)")


section("STEP 3 — Preprocessing & Feature Engineering")

df2 = df.copy()
df2 = df2.drop(columns=["Item_No"])

# Encode Operator: Op-1 → 0, Op-2 → 1 ... Op-20 → 19
le = LabelEncoder()
df2["Operator"] = le.fit_transform(df2["Operator"])
print(f"  Operator encoded (sample): Op-1={le.transform(['Op-1'])[0]}, "
      f"Op-20={le.transform(['Op-20'])[0]}")

# Engineered features
df2["Volume"]             = df2["Length"] * df2["Width"] * df2["Height"]
df2["Length_Width_Ratio"] = df2["Length"] / df2["Width"]
df2["Height_Width_Ratio"] = df2["Height"] / df2["Width"]
df2["LW_deviation"]       = (abs(df2["Length"] - df2["Length"].mean()) +
                              abs(df2["Width"]  - df2["Width"].mean()))
df2["dim_range"]          = (df2[["Length","Width","Height"]].max(axis=1) -
                              df2[["Length","Width","Height"]].min(axis=1))

print("  Engineered: Volume, Length_Width_Ratio, Height_Width_Ratio, "
      "LW_deviation, dim_range")

X = df2.drop(columns=["passed"])
y = df2["passed"].astype(int)

scaler   = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n  Features : {X.columns.tolist()}")
print(f"  Train set: {X_train.shape[0]} rows  |  Test set: {X_test.shape[0]} rows")


section("STEP 4 — SMOTE (Balance Training Set)")

minority_count = pd.Series(y_train).value_counts().min()
k = min(5, minority_count - 1)   # k_neighbors must be < minority count
print(f"  Before SMOTE: {dict(pd.Series(y_train).value_counts())}")
X_train_sm, y_train_sm = SMOTE(random_state=RANDOM_STATE,
                                k_neighbors=k).fit_resample(X_train, y_train)
print(f"  After  SMOTE: {dict(pd.Series(y_train_sm).value_counts())}")


section("STEP 5 — Train XGBoost")

# scale_pos_weight gives extra weight to inconsistent class as backup
neg, pos = np.bincount(y_train_sm)
spw = neg / pos

model = XGBClassifier(
    n_estimators     = 200,
    max_depth        = 5,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = spw,
    eval_metric      = "logloss",
    random_state     = RANDOM_STATE,
    n_jobs           = -1
)

model.fit(
    X_train_sm, y_train_sm,
    eval_set=[(X_test, y_test)],
    verbose=50
)
print(f"\n  scale_pos_weight used : {spw:.3f}")
print("  Training complete.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_f1 = cross_val_score(model, X_train_sm, y_train_sm,
                         cv=cv, scoring="f1_weighted", n_jobs=-1)
print(f"\n  5-Fold CV F1 (weighted): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")


section("STEP 6 — Evaluation Metrics")

y_pred = model.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)

acc     = accuracy_score(y_test, y_pred)
prec_w  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec_w   = recall_score   (y_test, y_pred, average="weighted", zero_division=0)
f1_w    = f1_score       (y_test, y_pred, average="weighted", zero_division=0)

prec_inc = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
rec_inc  = recall_score   (y_test, y_pred, pos_label=0, zero_division=0)
f1_inc   = f1_score       (y_test, y_pred, pos_label=0, zero_division=0)

prec_con = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec_con  = recall_score   (y_test, y_pred, pos_label=1, zero_division=0)
f1_con   = f1_score       (y_test, y_pred, pos_label=1, zero_division=0)

print(f"""
  ┌───────────────────────────────────────────────────┐
  │               EVALUATION RESULTS                  │
  ├──────────────────────────────────┬────────────────┤
  │ Metric                           │     Value      │
  ├──────────────────────────────────┼────────────────┤
  │ Accuracy  (overall)              │    {acc:.4f}      │
  │ Precision (weighted avg)         │    {prec_w:.4f}      │
  │ Recall    (weighted avg)         │    {rec_w:.4f}      │
  │ F1-Score  (weighted avg)         │    {f1_w:.4f}      │
  ├──────────────────────────────────┼────────────────┤
  │ Precision — Inconsistent class   │    {prec_inc:.4f}      │
  │ Recall    — Inconsistent class   │    {rec_inc:.4f}      │
  │ F1-Score  — Inconsistent class   │    {f1_inc:.4f}      │
  ├──────────────────────────────────┼────────────────┤
  │ Precision — Consistent class     │    {prec_con:.4f}      │
  │ Recall    — Consistent class     │    {rec_con:.4f}      │
  │ F1-Score  — Consistent class     │    {f1_con:.4f}      │
  ├──────────────────────────────────┼────────────────┤
  │ 5-Fold CV F1 (weighted)          │  {cv_f1.mean():.4f}±{cv_f1.std():.4f} │
  └──────────────────────────────────┴────────────────┘
""")

print("  Full Classification Report:\n")
print(classification_report(y_test, y_pred,
      target_names=["Inconsistent","Consistent"], zero_division=0))

print(f"  Confusion Matrix:\n{cm}\n")
print(f"  True Negatives  (defects caught)        : {cm[0][0]}")
print(f"  False Positives (good parts rejected)   : {cm[0][1]}")
print(f"  False Negatives (defects that slipped)  : {cm[1][0]}  ← minimise")
print(f"  True Positives  (good parts passed)     : {cm[1][1]}")

pd.DataFrame({
    "Metric": ["Accuracy","Precision (weighted)","Recall (weighted)","F1 (weighted)",
               "Precision - Inconsistent","Recall - Inconsistent","F1 - Inconsistent",
               "Precision - Consistent","Recall - Consistent","F1 - Consistent",
               "CV F1 Mean","CV F1 Std"],
    "Value":  [acc, prec_w, rec_w, f1_w,
               prec_inc, rec_inc, f1_inc,
               prec_con, rec_con, f1_con,
               cv_f1.mean(), cv_f1.std()]
}).to_csv(f"{OUTPUT_DIR}/metrics.csv", index=False)
print(f"\n  [SAVED] {OUTPUT_DIR}/metrics.csv")

section("STEP 7 — Generating Plots")

# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(6,5))
ConfusionMatrixDisplay(cm, display_labels=["Inconsistent","Consistent"]).plot(
    ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost — Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_confusion_matrix.png", dpi=150); plt.close()
print(f"  [SAVED] {OUTPUT_DIR}/01_confusion_matrix.png")

# 2. Metrics Bar Chart
fig, ax = plt.subplots(figsize=(11,5))
labels = ["Accuracy","Prec\n(Inconsist.)","Recall\n(Inconsist.)","F1\n(Inconsist.)",
          "Prec\n(Consist.)","Recall\n(Consist.)","F1\n(Consist.)"]
values = [acc, prec_inc, rec_inc, f1_inc, prec_con, rec_con, f1_con]
colors = ["#2c3e50","#e74c3c","#e74c3c","#e74c3c","#27ae60","#27ae60","#27ae60"]
bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.7, width=0.55)
for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
            f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)
ax.set_title("XGBoost — All Classification Metrics", fontsize=13, fontweight="bold")
ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#e74c3c", label="Inconsistent class"),
                   Patch(color="#27ae60", label="Consistent class"),
                   Patch(color="#2c3e50", label="Overall Accuracy")], loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_metrics_chart.png", dpi=150); plt.close()
print(f"  [SAVED] {OUTPUT_DIR}/02_metrics_chart.png")

# 3. Feature Importances
fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10,5))
fi.plot(kind="bar", ax=ax, color="#2980b9", edgecolor="black")
ax.set_title("XGBoost — Feature Importances", fontsize=13, fontweight="bold")
ax.set_ylabel("Importance Score")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_feature_importances.png", dpi=150); plt.close()
print(f"  [SAVED] {OUTPUT_DIR}/03_feature_importances.png")

# 4. Training Loss Curve
evals = model.evals_result()
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(evals["validation_0"]["logloss"], color="#e74c3c", linewidth=2)
ax.set_title("XGBoost — Validation Log-Loss (Convergence)", fontsize=12, fontweight="bold")
ax.set_xlabel("Boosting Round"); ax.set_ylabel("Log-Loss"); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_training_loss.png", dpi=150); plt.close()
print(f"  [SAVED] {OUTPUT_DIR}/04_training_loss.png")

# 5. Operator Inconsistency Rate
X_test_copy              = X_test.copy()
X_test_copy["predicted"] = y_pred
X_test_copy["Operator"]  = df2.loc[X_test.index, "Operator"].values
X_test_copy["Op_Name"]   = le.inverse_transform(X_test_copy["Operator"].astype(int))
op_rate = (X_test_copy.groupby("Op_Name")["predicted"]
           .apply(lambda x: (x==0).sum()/len(x)*100)
           .sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(13,4))
bar_colors = ["#e74c3c" if v > op_rate.mean() else "#95a5a6" for v in op_rate]
op_rate.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="black")
ax.axhline(op_rate.mean(), color="navy", linestyle="--",
           label=f"Average: {op_rate.mean():.1f}%")
ax.set_title("Inconsistency Rate by Operator  (red = above average)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Inconsistency Rate (%)"); ax.set_xlabel("Operator")
ax.legend(); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_operator_inconsistency.png", dpi=150); plt.close()
print(f"  [SAVED] {OUTPUT_DIR}/05_operator_inconsistency.png")

# 6. Dimension Distributions by Class
fig, axes = plt.subplots(1, 3, figsize=(14,4))
for ax, col in zip(axes, dims):
    df[df["passed"]==0][col].hist(bins=20, alpha=0.65, ax=ax,
                                   color="#e74c3c", label="Inconsistent")
    df[df["passed"]==1][col].hist(bins=20, alpha=0.65, ax=ax,
                                   color="#27ae60", label="Consistent")
    lo, hi = bounds[col]
    ax.axvline(lo, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(hi, color="black", linestyle="--", linewidth=1.2, label="±2σ bound")
    ax.set_title(f"{col}"); ax.legend(fontsize=8)
plt.suptitle("Dimensional Distributions — Consistent vs Inconsistent",
             fontweight="bold", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_dimension_distributions.png", dpi=150); plt.close()
print(f"  [SAVED] {OUTPUT_DIR}/06_dimension_distributions.png")

# =============================================================================
# STEP 8 — MITIGATION REPORT
# =============================================================================
section("STEP 8 — Mitigation Report")

all_pred = model.predict(
    pd.DataFrame(scaler.transform(df2.drop(columns=["passed"])), columns=X.columns)
)
df_full              = df.copy()
df_full["predicted"] = all_pred
op_full = (df_full.groupby("Operator")["predicted"]
           .apply(lambda x: (x==0).sum()/len(x)*100)
           .sort_values(ascending=False))
high_risk = op_full[op_full > op_full.mean()].index.tolist()

report = f"""
==============================================================
  MITIGATION REPORT — PS-2 PARTS MANUFACTURING QUALITY
  XGBoost Dimensional Consistency Classifier
==============================================================

1. MODEL PARAMETERS
──────────────────────────────────────────────────────────────
  Algorithm        : XGBoost (Gradient Boosted Trees)
  n_estimators     : 200
  max_depth        : 5
  learning_rate    : 0.1
  subsample        : 0.8
  colsample_bytree : 0.8
  Imbalance fix    : SMOTE + scale_pos_weight={spw:.3f}
  Train/Test split : 80% / 20% (stratified)

2. TARGET LABEL DEFINITION
──────────────────────────────────────────────────────────────
  No pre-labelled target existed in the CSV.
  Labels derived using ±2σ manufacturing tolerance bounds:

  Length : [{bounds['Length'][0]:.3f}, {bounds['Length'][1]:.3f}] mm
  Width  : [{bounds['Width'][0]:.3f}, {bounds['Width'][1]:.3f}] mm
  Height : [{bounds['Height'][0]:.3f}, {bounds['Height'][1]:.3f}] mm

  Parts outside ANY bound → INCONSISTENT (0)
  Parts within ALL bounds → CONSISTENT (1)

  Consistent   : {vc.get(1,0)} parts ({vc.get(1,0)/len(df)*100:.1f}%)
  Inconsistent : {vc.get(0,0)} parts ({vc.get(0,0)/len(df)*100:.1f}%)

3. EVALUATION RESULTS
──────────────────────────────────────────────────────────────
  Accuracy  (overall)         : {acc:.4f}
  Precision (weighted avg)    : {prec_w:.4f}
  Recall    (weighted avg)    : {rec_w:.4f}
  F1-Score  (weighted avg)    : {f1_w:.4f}
  5-Fold CV F1                : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}

  Inconsistent class:
    Precision : {prec_inc:.4f}
    Recall    : {rec_inc:.4f}
    F1-Score  : {f1_inc:.4f}

  Consistent class:
    Precision : {prec_con:.4f}
    Recall    : {rec_con:.4f}
    F1-Score  : {f1_con:.4f}

4. CONFUSION MATRIX
──────────────────────────────────────────────────────────────
{cm}

  True Negatives  (defects caught)        : {cm[0][0]}
  False Positives (good parts rejected)   : {cm[0][1]}
  False Negatives (defects slipped past)  : {cm[1][0]}  ← most critical
  True Positives  (good parts passed)     : {cm[1][1]}

5. TOP FEATURES BY IMPORTANCE
──────────────────────────────────────────────────────────────
{fi.to_string()}

6. OPERATOR INCONSISTENCY RATES (full dataset)
──────────────────────────────────────────────────────────────
{op_full.round(1).to_string()}

  Above-average risk: {', '.join(high_risk)}

7. MITIGATION STRATEGIES
──────────────────────────────────────────────────────────────

  A. OPERATOR SKILL
     • Operators {', '.join(high_risk)} exceed average inconsistency.
       Immediate targeted re-training required.
     • Weekly precision skill assessments for bottom-quartile.
     • Pair low performers with top performers (buddy system).
     • Real-time quality score dashboard per operator per shift.

  B. MACHINE CALIBRATION
     • Daily  : Zero-point calibration on all CMM instruments.
     • Weekly : Full dimensional check vs master gauge.
     • Monthly: Certified metrologist full recalibration.
     • Flag machines >5% inconsistency for immediate service.
     • SPC (X-bar & R) charts on Length/Width/Height.
       Alert when readings drift beyond ±2σ.

  C. TOOL WEAR
     • Replace cutting tools when moving-average deviation
       exceeds 50% of the tolerance band.
     • Log every change: part ID, machine, operator, timestamp.
     • Build regression model on deviation vs tool age for
       proactive replacement scheduling.

  D. INLINE ML QUALITY GATE
     Stage        │ Action                        │ Threshold
     ─────────────┼───────────────────────────────┼──────────────
     Post-cut     │ Score every part (this model) │ pred=0 → hold
     Post-grind   │ 10% sample CMM                │ ±0.05 mm
     Final QC     │ 100% CMM on all flagged parts │ Per drawing
     Shipping     │ Certificate of Conformity     │ QC sign-off

  E. CONTINUOUS IMPROVEMENT (PDCA)
     PLAN  → Reduce inconsistency rate by 5% per month.
     DO    → Apply strategies A–D; log all changes.
     CHECK → Retrain model monthly on new production data.
     ACT   → Standardise successes; escalate to engineering
             for Design-for-Manufacturability (DFM) review.

==============================================================
"""

print(report)
with open(f"{OUTPUT_DIR}/mitigation_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print(f"  [SAVED] {OUTPUT_DIR}/mitigation_report.txt")

section("DONE — All outputs saved to /results/")
print("""
  results/
  ├── 01_confusion_matrix.png
  ├── 02_metrics_chart.png
  ├── 03_feature_importances.png
  ├── 04_training_loss.png
  ├── 05_operator_inconsistency.png
  ├── 06_dimension_distributions.png
  ├── metrics.csv
  └── mitigation_report.txt
""")
