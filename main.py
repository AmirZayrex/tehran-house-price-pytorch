"""
Housing Price Prediction — main training pipeline.

Best config (found through experimentation):
    hidden_dims=[64, 32], dropout=0.1, lr=0.001, epochs=200, patience=30
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_data
from src.data.preprocessor import basic_process, basic_process_transform, remove_outliers_iqr
from src.features.features_engineering import add_numerical_features, add_transformation_features
from src.features.location_features import add_location_score
from src.data.formatter import to_pytorch
from src.models.pytorch_model import get_model
from src.training.trainer import train_pytorch_model, save_model
from src.evaluation.metrics import get_predictions, compute_price_metrics
from src.evaluation.visualizer import (
    plot_training_history,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_price_boxplot,
    plot_feature_distributions,
    plot_correlation_heatmap,
)


def main():
    print("=" * 60)
    print("Housing Price Prediction Pipeline (PyTorch)")
    print("=" * 60)

    # 1. Load
    df = load_data("Data/raw/HouseNew.csv")

    # 2. Outlier removal
    df = remove_outliers_iqr(df, column="Price")
    print(f"Shape after outlier removal: {df.shape}")

    # 3. Log-transform target
    df = add_transformation_features(df)

    # 4. EDA (on full cleaned data, before split)
    plot_price_boxplot(df, column="Price")
    plot_feature_distributions(df, features=["Area", "Room", "Floor"])
    plot_correlation_heatmap(df, columns=["Area", "Room", "Floor", "Price", "LogPrice"])

    # 5. Train / test split
    X = df.drop(columns=["LogPrice"])
    y = df["LogPrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # 6. Preprocessing
    # Train: transform + clean (row removal allowed)
    # Test:  transform only   (no row removal — avoids distribution shift)
    train_df = X_train.copy()
    train_df["LogPrice"] = y_train
    train_df = basic_process(train_df)
    X_train = train_df.drop(columns=["LogPrice"])
    y_train = train_df["LogPrice"]
    X_test = basic_process_transform(X_test)

    # 7. Feature engineering
    X_train = add_numerical_features(X_train)
    X_test  = add_numerical_features(X_test)

    # 8. Location encoding (K-Fold OOF — no leakage)
    X_train_temp = X_train.copy()
    X_train_temp["LogPrice"] = y_train
    X_train_temp, location_dict = add_location_score(X_train_temp, k=20)
    X_train = X_train_temp.drop(columns=["LogPrice"])
    X_test["LocationScore"] = X_test["Address"].map(location_dict).fillna(y_train.mean())

    # 9. Drop non-numeric columns
    X_train = X_train.drop(columns=["Address", "Price"])
    X_test  = X_test.drop(columns=["Address", "Price"])
    print(f"Features: {X_train.columns.tolist()}")

    # 10. Standardization
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    # 11. DataLoaders
    train_loader, test_loader = to_pytorch(X_train, y_train, X_test, y_test, batch_size=64)

    # 12. Model
    model = get_model(input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout_rate=0.1)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 13. Training
    history, trained_model = train_pytorch_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=200,
        learning_rate=0.001,
        early_stopping_patience=30,
    )

    # 14. Save artifacts
    save_model(trained_model, "Artifacts/pytorch_model.pth")
    joblib.dump(scaler,        "Artifacts/scaler.pkl")
    joblib.dump(location_dict, "Artifacts/location_dict.pkl")
    print("Artifacts saved: pytorch_model.pth | scaler.pkl | location_dict.pkl")

    # 15. Training history
    plot_training_history(history, title_suffix="(PyTorch)")

    # 16. Evaluation
    print("=" * 60)
    preds_log, actuals_log = get_predictions(trained_model, test_loader)
    metrics = compute_price_metrics(preds_log, actuals_log)
    print(f"RMSE : {metrics['rmse']:,.0f} Toman")
    print(f"MAE  : {metrics['mae']:,.0f} Toman")
    print(f"Mean : {metrics['mean_price']:,.0f} Toman")
    print(f"RMSE%: {metrics['rmse_pct']:.2f}%")

    # 17. Prediction diagnostics
    plot_actual_vs_predicted(actuals_log, preds_log, title_suffix="(PyTorch)")
    plot_residuals(actuals_log, preds_log, title_suffix="(PyTorch)")


if __name__ == "__main__":
    main()