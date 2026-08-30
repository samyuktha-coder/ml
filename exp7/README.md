# Experiment 6 (Dept. Handout) — Dimensionality Reduction and Model Evaluation (With and Without PCA)

**Course:** ICS1512 — Machine Learning Laboratory

## Files

| File | Description |
|---|---|
| `Assignment7_PCA.ipynb` | Notebook containing `assn7_experiment()` |
| `requirements_Assignment7.txt` | Python dependencies |

No separate dataset file is needed — the notebook loads the Wisconsin Diagnostic
Breast Cancer Dataset directly from `sklearn.datasets.load_breast_cancer()`
(569 samples, 30 numeric features, target: Malignant/Benign).

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements_Assignment7.txt
jupyter notebook
```

## What the notebook does

`assn7_experiment(df, target_column)` is a single generic function that works on
**any** tabular classification dataset. Steps:

1. Preprocess — stratified 80/20 train/test split, `StandardScaler`
2. PCA — number of components chosen to explain 95% of variance, with a scree plot
3. Hyperparameter grid defined for each of 9 tunable models
4. For **each** of 10 models (SVM, Naive Bayes, KNN, Logistic Regression,
   Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Stacking):
   `GridSearchCV` tuning + 5-fold cross-validation, run **twice** — once on the
   original 30 features (No-PCA), once on the PCA-reduced feature space (With-PCA)
5. Stacking uses a fixed structure (SVM + Naive Bayes + Decision Tree base
   learners, Logistic Regression meta-learner) evaluated the same way
6. Summary table — fold-wise and average CV accuracy, test accuracy/F1, for
   every model under both settings
7. Confusion matrices + overlaid ROC curves for SVM and Stacking (No-PCA vs
   With-PCA)
8. Bar chart comparing average CV accuracy per model, No-PCA vs With-PCA

## Running it

Open `Assignment7_PCA.ipynb` and run all cells top to bottom. Note: this notebook
trains 10 models × 2 settings × grid search × 5-fold CV — it takes noticeably
longer to run than the earlier single-model experiments (several minutes).

## Output

PCA scree plot, best-hyperparameters table per model (No-PCA vs With-PCA), the
full fold-wise summary table, confusion matrices, ROC curves, and the final
accuracy-comparison bar chart across all 10 models.
