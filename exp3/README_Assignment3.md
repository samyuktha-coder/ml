# Experiment 3 — Linear & Regularized Regression

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

## Files

| File | Description |
|---|---|
| `Assignment3_Regression.ipynb` | Notebook containing `assn3_experiment()` |
| `Loan_Dataset.csv` | Dataset (target column: `Loan Amount Request (USD)`) |
| `requirements_Assignment3.txt` | Python dependencies |

Keep the notebook and the CSV in the **same folder** — the notebook reads the CSV
by filename from the current directory.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements_Assignment3.txt
jupyter notebook
```

## What the notebook does

`assn3_experiment(df, target_column)` is a single, fully self-contained generic
function that works on **any** dataset with a continuous target — just change the
CSV and target column name to reuse it elsewhere. Steps:

1. Clean "fake" missing values (e.g. `"?"` placeholders → real NaN)
2. Inspect the data
3. Handle missing values (numeric → median, categorical → mode)
4. One-hot encode categorical columns
5. EDA — target distribution, feature-vs-target scatter plots
6. Split features/target + train/test split
7. Standardize features with `StandardScaler`
8. Baseline models — Linear, Ridge, Lasso, Elastic Net
9. 5-fold cross-validation for all 4 models
10. GridSearchCV vs RandomizedSearchCV tuning (Ridge/Lasso/Elastic Net)
11. Test set performance — tuned models
12. Predicted-vs-actual + residual plot (best model by R²)
13. Training vs validation error plot
14. Coefficient comparison plot

## Running it

Open `Assignment3_Regression.ipynb` and run all cells top to bottom.

## Output

CV table, tuning table, baseline and tuned test-set performance tables, plus the
four required diagnostic plots (predicted-vs-actual, residual, train-vs-validation
error, coefficient comparison).
