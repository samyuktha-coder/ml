"""
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 5: Decision Tree and Random Forest - A Comparative Classification Study
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), 569 samples, 30 features
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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
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
# 1. LOAD DATA  (WDBC ships with scikit-learn, identical to the UCI/Kaggle copy)
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
# target: 0 = malignant, 1 = benign in sklearn's encoding -> relabel so 1 = Malignant (positive class)
y = pd.Series(1 - data.target, name="diagnosis")  # 1 = Malignant, 0 = Benign
class_names = ["Benign", "Malignant"]

print(f"Loaded WDBC: {X.shape[0]} samples, {X.shape[1]} features")
missing = X.isnull().sum().sum()
duplicates = X.duplicated().sum()
print(f"Missing values: {missing} | Duplicate rows: {duplicates}")

dataset_info = {
    "dataset_name": "Wisconsin Diagnostic Breast Cancer (WDBC)",
    "dataset_source": "UCI Machine Learning Repository (via scikit-learn built-in loader)",
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "n_classes": 2,
    "missing_values": int(missing),
    "duplicates": int(duplicates),
    "class_counts": {"Benign": int((y == 0).sum()), "Malignant": int((y == 1).sum())},
}

# ---------------------------------------------------------------------------
# 2. SPLIT (80-20, stratified). No scaling needed for tree-based models.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
dataset_info["train_size"] = int(X_train.shape[0])
dataset_info["test_size"] = int(X_test.shape[0])
dataset_info["train_test_split"] = "80% / 20% (stratified)"
print(dataset_info)

# ---------------------------------------------------------------------------
# 3. EDA
# ---------------------------------------------------------------------------
plt.figure(figsize=(5, 4))
ax = sns.countplot(x=y, hue=y, palette=["#4C72B0", "#C44E52"], legend=False)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign (0)", "Malignant (1)"])
plt.title("Class Distribution"); plt.ylabel("Count"); plt.xlabel("")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(FIG / "class_distribution.png")
plt.close()

corr = X.corr()
top_feats = pd.concat([X, y], axis=1).corr()["diagnosis"].abs().sort_values(ascending=False).index[1:16]
plt.figure(figsize=(10, 8))
sns.heatmap(pd.concat([X[top_feats], y], axis=1).corr(), cmap="coolwarm", annot=True,
            fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix (Top 15 Features Most Correlated with Diagnosis)")
plt.tight_layout()
plt.savefig(FIG / "correlation_matrix.png")
plt.close()

key_feats = ["mean radius", "mean concave points", "worst area", "worst texture"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.ravel(), key_feats):
    sns.boxplot(x=y, y=X[feat], hue=y, palette=["#4C72B0", "#C44E52"], legend=False, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Malignant"])
    ax.set_title(feat); ax.set_xlabel("")
plt.suptitle("Feature Distributions by Class")
plt.tight_layout()
plt.savefig(FIG / "feature_distributions.png")
plt.close()

print("EDA plots saved.")

# ---------------------------------------------------------------------------
# 4. DECISION TREE - baseline + hyperparameter search (GridSearchCV, 5-fold)
# ---------------------------------------------------------------------------
dt_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    dt_param_grid, scoring="accuracy", cv=5, n_jobs=-1
)
t0 = time.time()
dt_grid.fit(X_train, y_train)
dt_search_time = time.time() - t0
dt_best = dt_grid.best_estimator_
print(f"Decision Tree best params: {dt_grid.best_params_}, CV acc: {dt_grid.best_score_:.4f}")

# Collapse full grid results to (criterion, max_depth) view for the report table
dt_cv_df = pd.DataFrame(dt_grid.cv_results_)
dt_cv_df["param_max_depth"] = dt_cv_df["param_max_depth"].apply(lambda v: "None" if v is None else v)
dt_depth_summary = (
    dt_cv_df.groupby(["param_criterion", "param_max_depth"])
    .agg(mean_acc=("mean_test_score", "max"))
    .reset_index()
    .sort_values("mean_acc", ascending=False)
)
dt_depth_summary.to_csv(RES / "dt_cv_by_depth.csv", index=False)

# F1 for the same grid (rerun scoring="f1" quickly via cross_val_score per best-per-depth combo)
dt_f1_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    dt_param_grid, scoring="f1", cv=5, n_jobs=-1
)
dt_f1_grid.fit(X_train, y_train)
dt_f1_df = pd.DataFrame(dt_f1_grid.cv_results_)
dt_f1_df["param_max_depth"] = dt_f1_df["param_max_depth"].apply(lambda v: "None" if v is None else v)

# ---------------------------------------------------------------------------
# 5. Overfitting study: train/test accuracy vs max_depth (single-tree, default other params)
# ---------------------------------------------------------------------------
depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, None]
train_accs, test_accs = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    m.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, m.predict(X_train)))
    test_accs.append(accuracy_score(y_test, m.predict(X_test)))
depth_labels = [str(d) if d is not None else "None" for d in depths]

plt.figure(figsize=(7, 5))
plt.plot(depth_labels, train_accs, "o-", label="Training Accuracy", color="#4C72B0")
plt.plot(depth_labels, test_accs, "o-", label="Test Accuracy", color="#C44E52")
plt.xlabel("max_depth"); plt.ylabel("Accuracy")
plt.title("Decision Tree: Overfitting vs. Tree Depth")
plt.legend(); plt.tight_layout()
plt.savefig(FIG / "dt_overfitting_depth.png")
plt.close()

# ---------------------------------------------------------------------------
# 6. RANDOM FOREST - hyperparameter search (GridSearchCV, 5-fold)
# ---------------------------------------------------------------------------
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, None],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    rf_param_grid, scoring="accuracy", cv=5, n_jobs=-1
)
t0 = time.time()
rf_grid.fit(X_train, y_train)
rf_search_time = time.time() - t0
rf_best = rf_grid.best_estimator_
print(f"Random Forest best params: {rf_grid.best_params_}, CV acc: {rf_grid.best_score_:.4f}")

rf_cv_df = pd.DataFrame(rf_grid.cv_results_)
rf_cv_df["param_max_depth"] = rf_cv_df["param_max_depth"].apply(lambda v: "None" if v is None else v)
rf_summary = (
    rf_cv_df.groupby(["param_n_estimators", "param_max_depth", "param_max_features"])
    .agg(mean_acc=("mean_test_score", "max"))
    .reset_index()
    .sort_values("mean_acc", ascending=False)
)
rf_summary.to_csv(RES / "rf_cv_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 7. FINAL FIT + EVALUATION (single fit each, timed)
# ---------------------------------------------------------------------------
t0 = time.time()
dt_final = DecisionTreeClassifier(random_state=RANDOM_STATE, **dt_grid.best_params_)
dt_final.fit(X_train, y_train)
dt_fit_time = time.time() - t0
dt_pred = dt_final.predict(X_test)
dt_proba = dt_final.predict_proba(X_test)[:, 1]

t0 = time.time()
rf_final = RandomForestClassifier(random_state=RANDOM_STATE, **rf_grid.best_params_)
rf_final.fit(X_train, y_train)
rf_fit_time = time.time() - t0
rf_pred = rf_final.predict(X_test)
rf_proba = rf_final.predict_proba(X_test)[:, 1]

def metrics_dict(y_true, y_pred, y_proba, fit_time):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "training_time": fit_time,
    }

dt_metrics = metrics_dict(y_test, dt_pred, dt_proba, dt_fit_time)
rf_metrics = metrics_dict(y_test, rf_pred, rf_proba, rf_fit_time)
print("Decision Tree metrics:", dt_metrics)
print("Random Forest metrics:", rf_metrics)

# ---------------------------------------------------------------------------
# 8. PLOTS: confusion matrices, ROC, tree viz, feature importance
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ConfusionMatrixDisplay(confusion_matrix(y_test, dt_pred), display_labels=class_names).plot(
    ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Decision Tree (Tuned)")
ConfusionMatrixDisplay(confusion_matrix(y_test, rf_pred), display_labels=class_names).plot(
    ax=axes[1], cmap="Greens", colorbar=False)
axes[1].set_title("Random Forest (Tuned)")
plt.tight_layout()
plt.savefig(FIG / "confusion_matrices.png")
plt.close()

plt.figure(figsize=(6, 5))
for name, proba, color in [("Decision Tree", dt_proba, "#4C72B0"), ("Random Forest", rf_proba, "#55A868")]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG / "roc_curve.png")
plt.close()

plt.figure(figsize=(16, 8))
plot_tree(dt_final, max_depth=3, feature_names=X.columns, class_names=class_names,
          filled=True, fontsize=8, proportion=True)
plt.title(f"Decision Tree Structure (top 3 levels shown; full depth={dt_final.get_depth()})")
plt.tight_layout()
plt.savefig(FIG / "tree_structure.png")
plt.close()

dt_imp = pd.Series(dt_final.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
rf_imp = pd.Series(rf_final.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
axes[0].barh(dt_imp.index[::-1], dt_imp.values[::-1], color="#4C72B0")
axes[0].set_title("Decision Tree - Top 15 Feature Importances")
axes[1].barh(rf_imp.index[::-1], rf_imp.values[::-1], color="#55A868")
axes[1].set_title("Random Forest - Top 15 Feature Importances")
plt.tight_layout()
plt.savefig(FIG / "feature_importance.png")
plt.close()

print("Result visualizations saved.")

# ---------------------------------------------------------------------------
# 9. 5-FOLD CROSS-VALIDATION COMPARISON (tuned models, on training set)
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
dt_cv_scores = cross_val_score(dt_final, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
rf_cv_scores = cross_val_score(rf_final, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)

cv_results = {
    "folds": [f"Fold {i+1}" for i in range(5)],
    "decision_tree": dt_cv_scores.tolist(),
    "random_forest": rf_cv_scores.tolist(),
    "dt_average": float(dt_cv_scores.mean()),
    "rf_average": float(rf_cv_scores.mean()),
    "dt_std": float(dt_cv_scores.std()),
    "rf_std": float(rf_cv_scores.std()),
}
print("5-Fold CV results:", cv_results)

plt.figure(figsize=(7, 5))
x = np.arange(5); w = 0.35
plt.bar(x - w/2, dt_cv_scores, width=w, label="Decision Tree", color="#4C72B0")
plt.bar(x + w/2, rf_cv_scores, width=w, label="Random Forest", color="#55A868")
plt.axhline(dt_cv_scores.mean(), color="#4C72B0", ls="--", lw=1, alpha=0.7)
plt.axhline(rf_cv_scores.mean(), color="#55A868", ls="--", lw=1, alpha=0.7)
plt.xticks(x, [f"Fold {i+1}" for i in range(5)])
plt.ylabel("Accuracy"); plt.title("5-Fold Cross-Validation Accuracy Comparison")
plt.legend(); plt.ylim(0.85, 1.0)
plt.tight_layout()
plt.savefig(FIG / "cv_comparison.png")
plt.close()

# ---------------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------------
all_results = {
    "dataset_info": dataset_info,
    "dt_best_params": {k: (v if v is not None else "None") for k, v in dt_grid.best_params_.items()},
    "dt_best_cv_accuracy": dt_grid.best_score_,
    "dt_metrics": dt_metrics,
    "dt_search_time": dt_search_time,
    "rf_best_params": {k: (v if v is not None else "None") for k, v in rf_grid.best_params_.items()},
    "rf_best_cv_accuracy": rf_grid.best_score_,
    "rf_metrics": rf_metrics,
    "rf_search_time": rf_search_time,
    "cv_results": cv_results,
    "dt_depth_overfit": {"depths": depth_labels, "train_acc": train_accs, "test_acc": test_accs},
    "sklearn_version": __import__("sklearn").__version__,
}
with open(RES / "results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

with open(RES / "classification_reports.txt", "w") as f:
    f.write("=== Decision Tree (Tuned) ===\n")
    f.write(classification_report(y_test, dt_pred, target_names=class_names))
    f.write("\n\n=== Random Forest (Tuned) ===\n")
    f.write(classification_report(y_test, rf_pred, target_names=class_names))

dt_depth_summary.head(10).to_string(open(RES / "dt_top10_configs.txt", "w"))
rf_summary.head(10).to_string(open(RES / "rf_top10_configs.txt", "w"))

print("\nAll results saved. Pipeline complete.")
