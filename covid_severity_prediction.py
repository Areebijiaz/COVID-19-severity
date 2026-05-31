# ============================================================
#  COVID Symptoms Severity Prediction — Kaggle Training Script
#  Targets: hospitalized | icu_admission | mortality
#  Output : results.zip  (models + metrics + plots)
# ============================================================

import os
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 0. Output folder
# ─────────────────────────────────────────
OUT = "output_files"
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
# On Kaggle: change the path below to /kaggle/input/<dataset-name>/your_file.csv
DATA_PATH = "/kaggle/input/covid-symptoms-severity/1777133026957_covid_symptoms_severity_prediction.csv"
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print(f"Dataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print("=" * 60)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# ─────────────────────────────────────────
# 2. EDA — save plots
# ─────────────────────────────────────────
TARGET_COLS = ["hospitalized", "icu_admission", "mortality"]
FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS]

# Target distribution
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, TARGET_COLS):
    counts = df[col].value_counts()
    ax.bar(["Negative", "Positive"], counts.values, color=["steelblue", "tomato"])
    ax.set_title(f"Target: {col}")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/target_distribution.png", dpi=120)
plt.close()
print("Saved: target_distribution.png")

# Correlation heatmap (numeric only)
num_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.4)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=120)
plt.close()
print("Saved: correlation_heatmap.png")

# Age distribution by gender
fig, ax = plt.subplots(figsize=(8, 4))
for gender, grp in df.groupby("gender"):
    ax.hist(grp["age"], bins=20, alpha=0.6, label=gender)
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.set_title("Age Distribution by Gender")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/age_distribution.png", dpi=120)
plt.close()
print("Saved: age_distribution.png")

# ─────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────
df_enc = df.copy()

# Encode categorical columns
le_gender = LabelEncoder()
le_vacc   = LabelEncoder()
df_enc["gender"]             = le_gender.fit_transform(df_enc["gender"])
df_enc["vaccination_status"] = le_vacc.fit_transform(df_enc["vaccination_status"])

print("\nGender encoding    :", dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))
print("Vaccination encoding:", dict(zip(le_vacc.classes_,  le_vacc.transform(le_vacc.classes_))))

X = df_enc[FEATURE_COLS].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

# ─────────────────────────────────────────
# 4. Models dictionary
# ─────────────────────────────────────────
MODELS = {
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost"             : XGBClassifier(n_estimators=200, random_state=42,
                                          use_label_encoder=False, eval_metric="logloss",
                                          verbosity=0),
    "SVM"                 : SVC(probability=True, random_state=42),
}

# ─────────────────────────────────────────
# 5. Train & Evaluate — one loop per target
# ─────────────────────────────────────────
summary_rows = []

for target in TARGET_COLS:
    print("\n" + "=" * 60)
    print(f"TARGET: {target}")
    print("=" * 60)

    y = df_enc[target]

    # Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE for class imbalance
    sm = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    print(f"  After SMOTE — train size: {X_tr_sm.shape[0]}  |  class ratio: {y_tr_sm.value_counts().to_dict()}")

    best_auc   = -1
    best_name  = ""
    best_model = None

    # --- per-model evaluation ---
    for name, model in MODELS.items():
        model.fit(X_tr_sm, y_tr_sm)
        y_pred      = model.predict(X_te)
        y_proba     = model.predict_proba(X_te)[:, 1]

        acc  = accuracy_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred, zero_division=0)
        auc  = roc_auc_score(y_te, y_proba)
        cv   = cross_val_score(model, X_scaled, y, cv=StratifiedKFold(5),
                                scoring="roc_auc", n_jobs=-1).mean()

        print(f"\n  [{name}]")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    F1 Score : {f1:.4f}")
        print(f"    ROC-AUC  : {auc:.4f}  (CV mean: {cv:.4f})")
        print(classification_report(y_te, y_pred, target_names=["Negative", "Positive"],
                                    zero_division=0))

        summary_rows.append({
            "Target"  : target,
            "Model"   : name,
            "Accuracy": round(acc, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC" : round(auc, 4),
            "CV AUC"  : round(cv, 4),
        })

        if auc > best_auc:
            best_auc   = auc
            best_name  = name
            best_model = model

    print(f"\n  ★ Best model for [{target}]: {best_name}  (AUC={best_auc:.4f})")

    # ── Confusion matrix for best model ──
    y_pred_best = best_model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred_best)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    ax.set_title(f"Confusion Matrix\n{target} — {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fname = f"{OUT}/cm_{target}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: cm_{target}.png")

    # ── ROC curve for best model ──
    y_proba_best = best_model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, y_proba_best)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {best_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {target}\n({best_name})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fname = f"{OUT}/roc_{target}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Saved: roc_{target}.png")

    # ── Feature importance (tree-based) ──
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
        importances = importances.sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        importances.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"Feature Importance — {target}\n({best_name})")
        ax.set_ylabel("Importance")
        ax.set_xlabel("Feature")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fname = f"{OUT}/feature_importance_{target}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  Saved: feature_importance_{target}.png")

# ─────────────────────────────────────────
# 6. Summary table
# ─────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{OUT}/model_summary.csv", index=False)
print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

# ── Summary bar chart ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
for ax, metric in zip(axes, metrics):
    pivot = summary_df.pivot(index="Model", columns="Target", values=metric)
    pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", width=0.7)
    ax.set_title(f"{metric} by Model & Target")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/summary_metrics.png", dpi=120)
plt.close()
print("Saved: summary_metrics.png")

# ─────────────────────────────────────────
# 7. Pack everything into results.zip
# ─────────────────────────────────────────
ZIP_PATH = "results.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(OUT):
        for file in files:
            full_path = os.path.join(root, file)
            zf.write(full_path, arcname=file)

print("\n" + "=" * 60)
print(f"All outputs saved to: {ZIP_PATH}")
print("Contents:")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    for name in zf.namelist():
        print(f"  • {name}")
print("=" * 60)
print("DONE ✓")
