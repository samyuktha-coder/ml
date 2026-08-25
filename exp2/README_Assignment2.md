# Experiment 2 — Naive Bayes & K-Nearest Neighbors

**Course:** ICS1512 — Machine Learning Algorithms Laboratory

## Files

| File | Description |
|---|---|
| `Assignment2_NB_KNN.ipynb` | Notebook containing `assn1_preprocess()` (reused) and `assn2_experiment()` |
| `Spambase_Dataset.csv` | Dataset (target column: `spam`) |
| `requirements_Assignment2.txt` | Python dependencies |

Keep the notebook and the CSV in the **same folder** — the notebook reads the CSV
by filename from the current directory.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements_Assignment2.txt
jupyter notebook
```

## What the notebook does

`assn1_preprocess()` (Assignment 1's function) is included first because this
experiment needs a cleaned, scaled train/test split as input. The actual
Experiment 2 logic is in `assn2_experiment(X_train, X_test, y_train, y_test)`,
which works on **any** dataset's preprocessed split:

1. Naive Bayes — Gaussian, Multinomial, Bernoulli (with confusion matrices)
2. KNN across k = 1, 3, 5, 7, 9, 11 — accuracy-vs-k plot
3. Best KNN (highest-accuracy k)
4. GridSearchCV vs RandomizedSearchCV tuning (`n_neighbors`, `weights`)
5. KDTree vs BallTree comparison
6. 5-fold cross-validation (Naive Bayes + Best KNN)
7. Final ranked model comparison table (Accuracy, Precision, Recall, F1, ROC-AUC,
   Train/Predict Time)

## Running it

Open `Assignment2_NB_KNN.ipynb` and run all cells top to bottom.

## Output

Confusion matrices for every model, the KNN accuracy-vs-k plot, tuning/tree/CV
tables, and a final sorted comparison table across all 6 models trained.
