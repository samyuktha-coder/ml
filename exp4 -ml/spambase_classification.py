"""
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 4: Binary Classification using Linear and Kernel-Based Models
Dataset: Spambase (UCI / Kaggle)

Pipeline:
1. Load + preprocess data
2. EDA (class distribution, correlation, feature distributions)
3. Train/test split + standardization
4. Baseline Logistic Regression
5. Hyperparameter tuning - Logistic Regression (GridSearchCV)
6. SVM with 4 kernels (baseline)
7. Hyperparameter tuning - SVM (RandomizedSearchCV, since kernel search space is large)
8. Evaluation (accuracy, precision, recall, F1, ROC-AUC) + plots
9. 5-Fold Cross-Validation comparison
10. Dump all results to JSON/CSV for report generation
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE = Path(__file__).parent
FIG = BASE / "figures"
RES = BASE / "results"
FIG.mkdir(exist_ok=True, parents=True)
RES.mkdir(exist_ok=True, parents=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 130

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
COLUMN_NAMES = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!",
    "char_freq_$", "char_freq_#",
    "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total",
    "spam"
]

raw = pd.read_csv(BASE.parent / "spambase.csv")
raw.columns = COLUMN_NAMES
print(f"Loaded dataset: {raw.shape[0]} rows, {raw.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------
missing = raw.isnull().sum().sum()
duplicates = raw.duplicated().sum()
print(f"Missing values: {missing} | Duplicate rows: {duplicates}")

df = raw.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["spam"])
y = df["spam"]

dataset_info = {
    "dataset_name": "Spambase",
    "dataset_source": "UCI Machine Learning Repository / Kaggle (somesh24/spambase)",
    "n_samples_raw": int(raw.shape[0]),
    "n_samples_after_dedup": int(df.shape[0]),
    "duplicates_removed": int(duplicates),
    "n_features": int(X.shape[1]),
    "n_classes": int(y.nunique()),
    "missing_values": int(missing),
    "class_counts": y.value_counts().to_dict(),
}

# Train-test split (stratified, 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
dataset_info["train_size"] = int(X_train.shape[0])
dataset_info["test_size"] = int(X_test.shape[0])
dataset_info["train_test_split"] = "80% / 20% (stratified)"

# Feature scaling (fit only on train, apply to test -- prevents leakage)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("Preprocessing complete.")
print(dataset_info)

# ---------------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------------
# 3a. Class distribution
plt.figure(figsize=(5, 4))
ax = sns.countplot(x=y, hue=y, palette=["#4C72B0", "#DD8452"], legend=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Ham (0)", "Spam (1)"])
plt.title("Class Distribution")
plt.ylabel("Count")
plt.xlabel("")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(FIG / "class_distribution.png")
plt.close()

# 3b. Statistical summary
summary_stats = df.describe().T
summary_stats.to_csv(RES / "statistical_summary.csv")

# 3c. Correlation matrix (top correlated features with target, for readability)
corr = df.corr()
top_corr_features = corr["spam"].abs().sort_values(ascending=False).index[1:16]
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_corr_features.tolist() + ["spam"]].corr(), cmap="coolwarm",
            annot=True, fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix (Top 15 Features Most Correlated with Target)")
plt.tight_layout()
plt.savefig(FIG / "correlation_matrix.png")
plt.close()

# 3d. Feature distributions (a representative few, spam vs ham)
key_feats = ["word_freq_free", "word_freq_your", "char_freq_!", "capital_run_length_average"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.ravel(), key_feats):
    sns.boxplot(x=y, y=np.log1p(df[feat]), hue=y, palette=["#4C72B0", "#DD8452"],
                legend=False, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Ham", "Spam"])
    ax.set_title(feat)
    ax.set_ylabel("log(1+value)")
    ax.set_xlabel("")
plt.suptitle("Feature Distributions by Class (log-scaled)")
plt.tight_layout()
plt.savefig(FIG / "feature_distributions.png")
plt.close()

print("EDA plots saved.")

# ---------------------------------------------------------------------------
# 4. BASELINE LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
t0 = time.time()
lr_base = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
lr_base.fit(X_train_s, y_train)
lr_base_time = time.time() - t0
lr_base_pred = lr_base.predict(X_test_s)

# ---------------------------------------------------------------------------
# 5. LOGISTIC REGRESSION - GRID SEARCH
# ---------------------------------------------------------------------------
lr_param_grid = [
    {"penalty": ["l1"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "saga"]},
    {"penalty": ["l2"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear", "saga"]},
]
lr_grid = GridSearchCV(
    LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    lr_param_grid, scoring="accuracy", cv=5, n_jobs=-1
)
t0 = time.time()
lr_grid.fit(X_train_s, y_train)
lr_grid_time = time.time() - t0
lr_best = lr_grid.best_estimator_
lr_best_pred = lr_best.predict(X_test_s)
lr_best_proba = lr_best.predict_proba(X_test_s)[:, 1]

print(f"Logistic Regression best params: {lr_grid.best_params_}, CV acc: {lr_grid.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 6. SVM - BASELINE ACROSS KERNELS
# ---------------------------------------------------------------------------
kernels = ["linear", "poly", "rbf", "sigmoid"]
svm_kernel_results = {}
svm_models = {}
for k in kernels:
    t0 = time.time()
    m = SVC(kernel=k, random_state=RANDOM_STATE)  # probability=False -> much faster fitting
    m.fit(X_train_s, y_train)
    ttime = time.time() - t0
    pred = m.predict(X_test_s)
    svm_kernel_results[k] = {
        "accuracy": accuracy_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "training_time": ttime,
    }
    svm_models[k] = m
    print(f"SVM ({k}): acc={svm_kernel_results[k]['accuracy']:.4f}, "
          f"f1={svm_kernel_results[k]['f1']:.4f}, time={ttime:.2f}s")

best_kernel = max(svm_kernel_results, key=lambda k: svm_kernel_results[k]["accuracy"])
print(f"Best baseline kernel: {best_kernel}")

# ---------------------------------------------------------------------------
# 7. SVM - RANDOMIZED SEARCH (search space is large -> randomized, per manual note)
# ---------------------------------------------------------------------------
svm_param_dist = [
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    {"kernel": ["sigmoid"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    {"kernel": ["poly"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"], "degree": [2, 3, 4]},
]
svm_random = RandomizedSearchCV(
    SVC(random_state=RANDOM_STATE),  # probability=False during search: much faster, scoring=accuracy doesn't need it
    param_distributions=svm_param_dist, n_iter=20, scoring="accuracy",
    cv=5, n_jobs=-1, random_state=RANDOM_STATE
)
t0 = time.time()
svm_random.fit(X_train_s, y_train)
svm_random_time = time.time() - t0

# Refit best-params model once with probability=True (needed only for ROC/PR curves)
svm_best = SVC(probability=True, random_state=RANDOM_STATE, **svm_random.best_params_)
svm_best.fit(X_train_s, y_train)
svm_best_pred = svm_best.predict(X_test_s)
svm_best_proba = svm_best.predict_proba(X_test_s)[:, 1]

print(f"SVM best params: {svm_random.best_params_}, CV acc: {svm_random.best_score_:.4f}")

# ---------------------------------------------------------------------------
# 8. EVALUATION METRICS FOR FINAL TUNED MODELS
# ---------------------------------------------------------------------------
def metrics_dict(y_true, y_pred, y_proba, train_time):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "training_time": train_time,
    }

lr_final_metrics = metrics_dict(y_test, lr_best_pred, lr_best_proba, lr_grid_time)
svm_final_metrics = metrics_dict(y_test, svm_best_pred, svm_best_proba, svm_random_time)
lr_baseline_metrics = metrics_dict(y_test, lr_base_pred, lr_base.predict_proba(X_test_s)[:, 1], lr_base_time)

print("Final LR metrics:", lr_final_metrics)
print("Final SVM metrics:", svm_final_metrics)

# ---------------------------------------------------------------------------
# 9. PLOTS - Confusion Matrices, ROC, PR, Learning Curve, Feature Importance
# ---------------------------------------------------------------------------
# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ConfusionMatrixDisplay(confusion_matrix(y_test, lr_best_pred),
                        display_labels=["Ham", "Spam"]).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Logistic Regression (Tuned)")
ConfusionMatrixDisplay(confusion_matrix(y_test, svm_best_pred),
                        display_labels=["Ham", "Spam"]).plot(ax=axes[1], cmap="Oranges", colorbar=False)
axes[1].set_title(f"SVM (Tuned, {svm_random.best_params_['kernel']} kernel)")
plt.tight_layout()
plt.savefig(FIG / "confusion_matrices.png")
plt.close()

# ROC curves
plt.figure(figsize=(6, 5))
for name, proba, color in [("Logistic Regression", lr_best_proba, "#4C72B0"),
                            ("SVM", svm_best_proba, "#DD8452")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG / "roc_curve.png")
plt.close()

# Precision-Recall curves
plt.figure(figsize=(6, 5))
for name, proba, color in [("Logistic Regression", lr_best_proba, "#4C72B0"),
                            ("SVM", svm_best_proba, "#DD8452")]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    plt.plot(rec, prec, label=name, color=color, lw=2)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve"); plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(FIG / "pr_curve.png")
plt.close()

# Learning curve (tuned LR, cheapest to compute reliably)
train_sizes, train_scores, val_scores = learning_curve(
    lr_best, X_train_s, y_train, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1, random_state=RANDOM_STATE
)
plt.figure(figsize=(6.5, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score", color="#4C72B0")
plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Cross-val score", color="#DD8452")
plt.fill_between(train_sizes, train_scores.mean(1)-train_scores.std(1),
                  train_scores.mean(1)+train_scores.std(1), alpha=0.15, color="#4C72B0")
plt.fill_between(train_sizes, val_scores.mean(1)-val_scores.std(1),
                  val_scores.mean(1)+val_scores.std(1), alpha=0.15, color="#DD8452")
plt.xlabel("Training Set Size"); plt.ylabel("Accuracy")
plt.title("Learning Curve - Logistic Regression (Tuned)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG / "learning_curve.png")
plt.close()

# Feature importance (LR coefficients - magnitude)
coefs = pd.Series(lr_best.coef_[0], index=X.columns).sort_values(key=abs, ascending=False).head(15)
plt.figure(figsize=(8, 6))
colors = ["#DD8452" if c > 0 else "#4C72B0" for c in coefs.values]
plt.barh(coefs.index[::-1], coefs.values[::-1], color=colors[::-1])
plt.xlabel("Coefficient Value (standardized features)")
plt.title("Top 15 Feature Importances - Logistic Regression\n(orange = pushes toward spam, blue = pushes toward ham)")
plt.tight_layout()
plt.savefig(FIG / "feature_importance.png")
plt.close()

print("Result visualizations saved.")

# ---------------------------------------------------------------------------
# 10. 5-FOLD CROSS-VALIDATION (tuned models, on full train set)
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
svm_best_nop = SVC(random_state=RANDOM_STATE, **svm_random.best_params_)  # no probability=True -> fast CV
lr_cv_scores = cross_val_score(lr_best, X_train_s, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
svm_cv_scores = cross_val_score(svm_best_nop, X_train_s, y_train, cv=skf, scoring="accuracy", n_jobs=-1)

cv_results = {
    "folds": [f"Fold {i+1}" for i in range(5)],
    "logistic_regression": lr_cv_scores.tolist(),
    "svm": svm_cv_scores.tolist(),
    "lr_average": float(lr_cv_scores.mean()),
    "svm_average": float(svm_cv_scores.mean()),
    "lr_std": float(lr_cv_scores.std()),
    "svm_std": float(svm_cv_scores.std()),
}
print("5-Fold CV results:", cv_results)

# CV comparison plot
plt.figure(figsize=(7, 5))
x = np.arange(5)
w = 0.35
plt.bar(x - w/2, lr_cv_scores, width=w, label="Logistic Regression", color="#4C72B0")
plt.bar(x + w/2, svm_cv_scores, width=w, label="SVM", color="#DD8452")
plt.axhline(lr_cv_scores.mean(), color="#4C72B0", ls="--", lw=1, alpha=0.7)
plt.axhline(svm_cv_scores.mean(), color="#DD8452", ls="--", lw=1, alpha=0.7)
plt.xticks(x, [f"Fold {i+1}" for i in range(5)])
plt.ylabel("Accuracy"); plt.title("5-Fold Cross-Validation Accuracy Comparison")
plt.legend()
plt.ylim(0.85, 1.0)
plt.tight_layout()
plt.savefig(FIG / "cv_comparison.png")
plt.close()

# ---------------------------------------------------------------------------
# SAVE ALL RESULTS
# ---------------------------------------------------------------------------
all_results = {
    "dataset_info": dataset_info,
    "lr_baseline_metrics": lr_baseline_metrics,
    "lr_best_params": lr_grid.best_params_,
    "lr_best_cv_accuracy": lr_grid.best_score_,
    "lr_final_metrics": lr_final_metrics,
    "svm_kernel_results": svm_kernel_results,
    "best_baseline_kernel": best_kernel,
    "svm_best_params": svm_random.best_params_,
    "svm_best_cv_accuracy": svm_random.best_score_,
    "svm_final_metrics": svm_final_metrics,
    "cv_results": cv_results,
    "sklearn_version": __import__("sklearn").__version__,
    "pandas_version": pd.__version__,
    "numpy_version": np.__version__,
}

with open(RES / "results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# Classification reports (text, for appendix)
with open(RES / "classification_reports.txt", "w") as f:
    f.write("=== Logistic Regression (Tuned) ===\n")
    f.write(classification_report(y_test, lr_best_pred, target_names=["Ham", "Spam"]))
    f.write("\n\n=== SVM (Tuned) ===\n")
    f.write(classification_report(y_test, svm_best_pred, target_names=["Ham", "Spam"]))

print("\nAll results saved to results/results.json")
print("Pipeline complete.")
