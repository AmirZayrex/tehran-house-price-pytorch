# Tehran House Price Prediction

End-to-end ML pipeline for predicting residential property prices in Tehran using a PyTorch neural network.

**Author:** AmirHooshang Amirjani  
**Repo:** [github.com/AmirZayrex/tehran-house-price-pytorch](https://github.com/AmirZayrex/tehran-house-price-pytorch)

---

## Overview

Most housing price models treat the entire market as a single distribution. This project challenges that assumption by building a clean, modular pipeline that is explicitly designed for **segmented multi-model architecture** — the natural next step when one model is not enough.

Current results with a single neural network:

| Metric | Value |
|--------|-------|
| RMSE | ~1.07B Toman |
| MAE | ~628M Toman |
| RMSE% of mean | 24.56% |

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── loader.py              # CSV loading with validation
│   │   ├── preprocessor.py        # Missing value handling, boolean conversion, outlier removal
│   │   └── formatter.py           # DataFrame → PyTorch DataLoader / NumPy
│   ├── features/
│   │   ├── features_engineering.py  # AreaPerRoom, Age, LogPrice
│   │   └── location_features.py     # Bayesian target encoding with K-Fold OOF
│   ├── models/
│   │   └── pytorch_model.py       # Configurable fully-connected network
│   ├── training/
│   │   └── trainer.py             # Training loop with early stopping
│   └── evaluation/
│       ├── metrics.py             # RMSE, MAE, RMSE% on actual price scale
│       └── visualizer.py          # Training history, residuals, correlation heatmap
├── Data/
│   └── raw/HouseNew.csv
├── Artifacts/                     # Saved model, scaler, location dict
├── app.py                         # Streamlit UI
└── main.py                        # Training pipeline entry point
```

---

## Key Engineering Decisions

### 1. Log Transform on Target
Tehran house prices are heavily right-skewed. Applying `log1p(Price)` reduces skewness, dampens the effect of ultra-luxury outliers, and makes the regression task significantly easier for the model. All metrics are converted back to actual Toman scale via `expm1`.

### 2. Bayesian Location Encoding with K-Fold OOF
Address is a high-cardinality categorical feature (~400+ unique neighborhoods). Instead of one-hot encoding (sparse, high-dimensional) or naive mean encoding (data leakage), this project uses **smoothed target encoding**:

```
score = (count × mean + k × global_mean) / (count + k)
```

- `k=20` pulls low-count neighborhoods toward the global mean, preventing unreliable scores from single-listing areas.
- **K-Fold out-of-fold** ensures each row's score is computed from other folds only — never from itself. This eliminates target leakage entirely.

### 3. Train / Test Asymmetry in Preprocessing
Row removal (e.g. `Room <= 0`) is applied **only to the training set**. The test set receives only transformations. This reflects real deployment conditions where invalid inputs must be handled gracefully rather than silently dropped.

### 4. Data Leakage Discovery
During development, `Price` was inadvertently left in the feature matrix alongside `LogPrice`. The model was effectively learning to predict `log(Price)` from `Price` directly — achieving near-zero test loss but zero generalization. Removing `Price` from features revealed the true model performance.

---

## Model Architecture

```
Input(9) → Linear(64) → ReLU → Dropout(0.1) → Linear(32) → ReLU → Dropout(0.1) → Linear(1)
```

Best hyperparameters found through experimentation:

| Parameter | Value |
|-----------|-------|
| hidden_dims | [64, 32] |
| dropout_rate | 0.1 |
| learning_rate | 0.001 |
| epochs | 200 |
| early_stopping_patience | 30 |
| batch_size | 64 |

---

## Why 24.56% RMSE?

A single model for the entire Tehran housing market has a hard ceiling. The market behavior of a 2B Toman apartment in Shahrak Gharb and a 50B Toman penthouse in Niavaran are fundamentally different — one model cannot capture both well.

The architecture is explicitly designed for the next step: **market segmentation**.

```
LogPrice < Q1          →  Low segment
Q1 ≤ LogPrice ≤ Q3    →  Core segment
Q3 < LogPrice ≤ Q85   →  Luxury segment
LogPrice > Q85         →  Super-Luxury segment
```

Each segment would receive its own model, routing predictions through a segment classifier at inference time. This is the planned next phase of the project.

---

## Setup

```bash
pip install torch pandas scikit-learn matplotlib seaborn streamlit joblib
```

**Train:**
```bash
python main.py
```

**Run UI:**
```bash
streamlit run app.py
```

---

## Features

| Feature | Description |
|---------|-------------|
| Area | Property size in m² |
| Room | Number of rooms |
| Floor | Floor number |
| Age | Years since construction |
| AreaPerRoom | Area / Room (space quality proxy) |
| LocationScore | Bayesian-smoothed neighborhood price score |
| Elevator | Binary amenity |
| Parking | Binary amenity |
| Warehouse | Binary amenity |
