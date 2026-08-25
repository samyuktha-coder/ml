# Experiment 1 — Data Preprocessing & Exploratory Data Analysis

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

## Files

| File | Description |
|---|---|
| `Assignment1_Preprocessing.ipynb` | Notebook containing `assn1_preprocess()` |
| `Spambase_Dataset.csv` | Dataset (target column: `spam`) |
| `requirements_Assignment1.txt` | Python dependencies |

Keep the notebook and the CSV in the **same folder** — the notebook reads the CSV
by filename from the current directory.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements_Assignment1.txt
jupyter notebook
```

## What the notebook does

`assn1_preprocess(df, target_column)` is a single generic function that works on
**any** tabular classification dataset — just change the CSV and target column
name to reuse it elsewhere. Steps:

1. Inspect the data (shape, dtypes, summary stats, missing values, duplicates)
2. Remove duplicate rows
3. Handle missing values (numeric → median, categorical → mode)
4. Split features / target
5. EDA plots — class distribution, histograms, boxplots, correlation heatmap
6. Scale features with `MinMaxScaler`
7. Stratified train/test split (80/20)

## Running it

Open `Assignment1_Preprocessing.ipynb` and run all cells top to bottom.

## Output

A dict containing `df`, `X`, `y`, `X_train`, `X_test`, `y_train`, `y_test`, and
`scaler`, plus the EDA plots rendered inline. This output feeds directly into
Experiment 2.
