# ICS1512 — Experiment 9: Perceptron vs. MLP for Handwritten English Character Recognition

Implements and compares two classifiers on the [English Handwritten Characters Dataset](https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset) (3410 images, 62 classes — digits 0–9, uppercase A–Z, lowercase a–z):

- **Model A — Perceptron Learning Algorithm (PLA):** one-vs-rest linear units with step activation, trained from scratch (no autograd).
- **Model B — Multi-Layer Perceptron (MLP):** fully-connected feed-forward network trained by back-propagation, tuned over architecture, activation, loss, optimizer, learning rate, batch size, dropout and L2 weight decay via random search + greedy stage-wise (coordinate) search.

The full write-up, tables and inferences are in the accompanying lab report (`report.tex` / `ML_LAB_EXP9_Report.pdf`).

## Repository contents

| File | Description |
|---|---|
| `MLLABEX9___1_.ipynb` | Main notebook — data loading, PLA, MLP, tuning, evaluation, plots |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU is optional — the notebook auto-detects CUDA and falls back to CPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

## Dataset

The notebook downloads the dataset automatically via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("dhruvildave/english-handwritten-characters-dataset")
```

This requires a Kaggle account and API credentials (`~/.kaggle/kaggle.json`, or the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables). See the [Kaggle API docs](https://github.com/Kaggle/kaggle-api#api-credentials) for setup.

If you already have the dataset locally, skip the `kagglehub` cell and point `--data_dir` (in the `main()` args cell) at the folder containing `english.csv` and the `Img/` directory.

## Running

Open and run all cells in order:

```bash
jupyter notebook MLLABEX9___1_.ipynb
```

Key stages, in order:

1. **Data loading & preprocessing** — grayscale → resize to 32×32 (bilinear) → scale to [0,1] and invert → flatten to 1024-d → stratified 70/15/15 train/val/test split (seed=42) → standardize using train-split statistics.
2. **Model A (PLA)** — trained for 100 epochs, `lr=0.1`, best-validation epoch retained.
3. **Model B (MLP) tuning** — 24-trial random search (30 epochs each) over the hyper-parameter space, followed by greedy stage-wise search (optimizer × lr, batch size, activation, loss, architecture/depth, dropout, weight decay), then a final 100-epoch run with best-validation checkpointing.
4. **Evaluation** — accuracy, macro/weighted precision/recall/F1, micro/macro one-vs-rest ROC-AUC, confusion matrices, for both models on the held-out test set.
5. **Plots** — all figures (sample grid, hyper-parameter impact, tuning heat-maps, convergence curves, confusion matrices, ROC curves, A/B comparison, overfitting study) are saved to `--out_dir` (default `/content/exp9_outputs`).

Default settings run the full budget described in the report (100 PLA epochs, 24 tuning trials, 100 final MLP epochs) and can take a while on CPU. Pass `--quick` in the `main()` args cell for a fast smoke-test run (10 PLA epochs, 4 trials, 15 final epochs) with much lower accuracy — use this only to check the pipeline runs end-to-end, not to reproduce the reported numbers.

## Output

All artifacts (plots, confusion-matrix CSVs, classification reports, and a text file of auto-generated observations) are written to the `--out_dir` passed to `main()`.

## Author

Samyuktha V — [github.com/samyuktha-coder/ml](https://github.com/samyuktha-coder/ml/tree/main)
