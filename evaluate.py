"""
Evaluation script — loads saved artifacts and runs all visualizations
without re-training the model.

Usage:
    python evaluate.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.data.loader import load_data
from src.data.preprocessor import basic_process, basic_process_transform, remove_outliers_iqr
from src.features.features_engineering import add_numerical_features, add_transformation_features
from src.features.location_features import add_location_score
from src.data.formatter import to_pytorch
from src.models.pytorch_model import get_model
from src.evaluation.metrics import get_predictions, compute_price_metrics
from src.evaluation.visualizer import (
    plot_training_history,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_price_boxplot,
    plot_feature_distributions,
    plot_correlation_heatmap,
)


def load_artifacts():
    """Load saved model, scaler, location dict, and training history."""
    scaler = joblib.load("Artifacts/scaler.pkl")
    location_dict = joblib.load("Artifacts/location_dict.pkl")

    with open("Artifacts/history.json") as f:
        history = json.load(f)

    model = get_model(input_dim=9, hidden_dims=[64, 32], dropout_rate=0.1)
    model.load_state_dict(torch.load("Artifacts/pytorch_model.pth", map_location="cpu"))
    model.eval()

    print("Artifacts loaded successfully")
    return model, scaler, location_dict, history


def build_test_loader(scaler, location_dict):
    """Rebuild the test set using the same pipeline as training."""
    df = load_data("Data/raw/HouseNew.csv")
    df = remove_outliers_iqr(df, column="Price")
    df = add_transformation_features(df)

    X = df.drop(columns=["LogPrice"])
    y = df["LogPrice"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test = basic_process_transform(X_test)
    X_test = add_numerical_features(X_test)
    X_test["LocationScore"] = X_test["Address"].map(location_dict).fillna(y.mean())
    X_test = X_test.drop(columns=["Address", "Price"])

    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    _, test_loader = to_pytorch(
        X_test, y_test,  # dummy train — only test_loader is used
        X_test, y_test,
        batch_size=64,
    )
    return test_loader, df


def main():
    print("=" * 60)
    print("Evaluation — Housing Price Prediction (PyTorch)")
    print("=" * 60)

    model, scaler, location_dict, history = load_artifacts()
    test_loader, df = build_test_loader(scaler, location_dict)

    # EDA
    plot_price_boxplot(df, column="Price")
    plot_feature_distributions(df, features=["Area", "Room", "Floor"])
    plot_correlation_heatmap(df, columns=["Area", "Room", "Floor", "Price", "LogPrice"])

    # Training history
    plot_training_history(history, title_suffix="(PyTorch)")

    # Metrics
    print("=" * 60)
    preds_log, actuals_log = get_predictions(model, test_loader)
    metrics = compute_price_metrics(preds_log, actuals_log)
    print(f"RMSE : {metrics['rmse']:,.0f} Toman")
    print(f"MAE  : {metrics['mae']:,.0f} Toman")
    print(f"Mean : {metrics['mean_price']:,.0f} Toman")
    print(f"RMSE%: {metrics['rmse_pct']:.2f}%")

    # Prediction diagnostics
    plot_actual_vs_predicted(actuals_log, preds_log, title_suffix="(PyTorch)")
    plot_residuals(actuals_log, preds_log, title_suffix="(PyTorch)")


if __name__ == "__main__":
    main()