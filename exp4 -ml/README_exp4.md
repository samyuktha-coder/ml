# Experiment 4 — Binary Classification using Linear and Kernel-Based Models

ICS1512 Machine Learning Algorithms Laboratory. Classifies emails as spam/ham
using Logistic Regression and SVM, with hyperparameter tuning and comparison.

## Dataset
Spambase (UCI / Kaggle `somesh24/spambase`) — 4601 samples, 57 features, binary
label (1 = spam, 0 = ham). `spambase.csv` included, no header row needed (script
assigns standard column names).

## Setup
```bash
pip install -r requirements_exp4.txt
```

## Run
```bash
python spambase_classification.py
```
Place `spambase.csv` one directory above the script (i.e. `../spambase.csv`
relative to the script), or edit the `pd.read_csv(...)` path at the top of the
file.

## Output
- `figures/` — class distribution, correlation matrix, feature distributions,
  confusion matrices, ROC curve, PR curve, learning curve, feature importance,
  CV comparison (9 PNGs)
- `results/results.json` — all metrics, best hyperparameters, CV scores
- `results/statistical_summary.csv` — full 57-feature descriptive stats
- `results/classification_reports.txt` — sklearn classification reports

## Pipeline
1. Load data, drop duplicates, check for missing values
2. 80/20 stratified train-test split + `StandardScaler`
3. EDA (class balance, correlation, feature distributions)
4. Baseline Logistic Regression
5. Logistic Regression tuning — `GridSearchCV` (L1/L2, C, solver)
6. SVM baseline across 4 kernels (linear, poly, RBF, sigmoid)
7. SVM tuning — `RandomizedSearchCV` (kernel, C, gamma, degree)
8. Evaluation — accuracy, precision, recall, F1, ROC-AUC
9. 5-fold cross-validation comparison

## Results Summary
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression (tuned) | 0.9371 | 0.9203 | 0.9774 |
| SVM (tuned, RBF) | 0.9335 | 0.9154 | 0.9776 |

Full writeup: `Experiment_4_Report.pdf`.
