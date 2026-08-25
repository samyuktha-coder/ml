# Experiment 5 — Decision Tree and Random Forest: A Comparative Classification Study

ICS1512 Machine Learning Algorithms Laboratory. Classifies breast masses as
Malignant/Benign using a Decision Tree and a Random Forest, with hyperparameter
tuning and comparison.

## Dataset
Wisconsin Diagnostic Breast Cancer (WDBC) — 569 samples, 30 features, binary
label (Malignant / Benign). Loaded directly via
`sklearn.datasets.load_breast_cancer()` — identical to the UCI archive copy,
no download needed. `wdbc_breast_cancer.csv` included as a standalone export
if you need a flat file instead.

## Setup
```bash
pip install -r requirements_exp5.txt
```

## Run
```bash
python decision_tree_random_forest.py
```
No external dataset file required — it loads from scikit-learn directly.

## Output
- `figures/` — class distribution, correlation matrix, feature distributions,
  overfitting-vs-depth curve, confusion matrices, ROC curve, tree structure,
  feature importance, CV comparison (9 PNGs)
- `results/results.json` — all metrics, best hyperparameters, CV scores
- `results/dt_cv_by_depth.csv`, `results/rf_cv_summary.csv` — full grid search results
- `results/classification_reports.txt` — sklearn classification reports

## Pipeline
1. Load data (label: 1 = Malignant, 0 = Benign), check missing/duplicates
2. 80/20 stratified train-test split (no scaling needed — tree-based models)
3. EDA (class balance, correlation, feature distributions)
4. Decision Tree tuning — `GridSearchCV` (criterion, max_depth, min_samples_split/leaf)
5. Overfitting study — train vs. test accuracy across increasing max_depth
6. Random Forest tuning — `GridSearchCV` (n_estimators, max_depth, max_features, bootstrap)
7. Evaluation — accuracy, precision, recall, F1, ROC-AUC
8. 5-fold cross-validation comparison

## Results Summary
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Decision Tree (tuned) | 0.9561 | 0.9367 | 0.9396 |
| Random Forest (tuned) | 0.9649 | 0.9500 | 0.9940 |

Best DT: entropy, max_depth=7, min_samples_split=5, min_samples_leaf=2 (94.73% CV acc)
Best RF: n_estimators=200, max_depth=10, max_features=log2, bootstrap=False (97.14% CV acc)

Full writeup: `Experiment_5_Report.pdf`.
