import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_history(history: dict, title_suffix: str = "") -> None:
    """Train vs Test Loss over epochs on a log scale."""
    train_loss = history["train_loss"]
    test_loss = history["test_loss"]
    epochs = range(1, len(train_loss) + 1)
    best_epoch = test_loss.index(min(test_loss)) + 1

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, color="steelblue", linewidth=2, label="Train Loss")
    plt.plot(epochs, test_loss, color="red", linestyle="--", linewidth=2, label="Test Loss")
    plt.axvline(best_epoch, color="black", linestyle=":", linewidth=1.5, label=f"Best Epoch ({best_epoch})")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"Training History {title_suffix}".strip())
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(actuals_log: np.ndarray, preds_log: np.ndarray, title_suffix: str = "") -> None:
    """Scatter plot of Actual vs Predicted on LogPrice scale."""
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals_log, preds_log, alpha=0.4, c=actuals_log, cmap="viridis")
    min_val = min(actuals_log.min(), preds_log.min())
    max_val = max(actuals_log.max(), preds_log.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2)
    plt.xlabel("Actual Log Price")
    plt.ylabel("Predicted Log Price")
    plt.title(f"Actual vs Predicted {title_suffix}".strip())
    plt.colorbar(label="Log Price")
    plt.tight_layout()
    plt.show()


def plot_residuals(actuals_log: np.ndarray, preds_log: np.ndarray, title_suffix: str = "") -> None:
    """Residuals vs Actual — shows where the model struggles."""
    residuals = actuals_log - preds_log
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals_log, residuals, alpha=0.4, c=np.abs(residuals), cmap="plasma")
    plt.axhline(0, color="black", linestyle="--", linewidth=2)
    plt.xlabel("Actual Log Price")
    plt.ylabel("Residual")
    plt.title(f"Residuals vs Actual {title_suffix}".strip())
    plt.colorbar(label="Absolute Residual")
    plt.tight_layout()
    plt.show()


def plot_price_boxplot(df: pd.DataFrame, column: str = "Price") -> None:
    """Boxplot for outlier inspection."""
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(df: pd.DataFrame, features: list[str]) -> None:
    """Histogram grid for numeric feature distributions."""
    n = len(features)
    plt.figure(figsize=(4 * n, 4))
    for i, feat in enumerate(features, 1):
        plt.subplot(1, n, i)
        sns.histplot(df[feat], bins=20, kde=True, color="skyblue")
        plt.title(f"{feat} Distribution")
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str]) -> None:
    """Correlation heatmap for selected columns."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()