# Experiment 6/7 — Bagging, Boosting, and Stacked Ensemble Models

**Course:** ICS1512 — Machine Learning Laboratory

## Files

| File | Description |
|---|---|
| `Assignment6_Ensemble.ipynb` | Notebook containing `assn6_experiment()` |
| `requirements_Assignment6.txt` | Python dependencies |

No separate dataset file is needed — the notebook loads the Wisconsin Diagnostic
Breast Cancer Dataset directly from `sklearn.datasets.load_breast_cancer()`
(569 samples, 30 numeric features, target: Malignant/Benign).

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements_Assignment6.txt
jupyter notebook
```

## What the notebook does

`assn6_experiment(df, target_column)` is a single generic function that works on
**any** tabular classification dataset — just pass a dataframe and target column
name to reuse it elsewhere. Steps:

1. Preprocess — stratified 80/20 train/test split, `StandardScaler`
2. EDA — class distribution, correlation heatmap, feature histograms
3. **Bagging** — `BaggingClassifier` (Decision Tree base estimator), tuned over
   `n_estimators`, `max_samples`, `max_features` via `GridSearchCV` (5-fold CV)
4. **Boosting** — `AdaBoostClassifier` and `GradientBoostingClassifier`, tuned over
   `n_estimators`, `learning_rate`, `max_depth`; the better-performing variant is
   carried forward
5. **Stacked Ensemble** — base learners SVM + Gaussian Naive Bayes + Decision Tree,
   meta-learner Logistic Regression (`StackingClassifier`)
6. Final test-set evaluation — Accuracy, Precision, Recall, F1, confusion matrices,
   ROC curves + AUC for all three ensemble models
7. Bias–variance check — train-vs-test accuracy bar chart

## Running it

Open `Assignment6_Ensemble.ipynb` and run all cells top to bottom.

## Output

Hyperparameter evaluation tables for Bagging/Boosting/Stacking, confusion matrices,
an overlaid ROC curve plot, the final performance comparison table, and a
train-vs-test accuracy chart for bias–variance discussion.
