import numpy as np
import torch
from torch.utils.data import DataLoader


def get_predictions(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Run model over a DataLoader and return (predictions, actuals) on LogPrice scale."""
    device = next(model.parameters()).device
    model.eval()

    all_preds, all_actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).squeeze(-1)
            all_preds.append(preds)
            all_actuals.append(y_batch.to(device))

    return torch.cat(all_preds).cpu().numpy(), torch.cat(all_actuals).cpu().numpy()


def compute_price_metrics(preds_log: np.ndarray, actuals_log: np.ndarray) -> dict:
    """Convert LogPrice predictions to actual Price scale and compute RMSE, MAE, RMSE%."""
    preds_price = np.expm1(preds_log)
    actuals_price = np.expm1(actuals_log)
    mean_price = actuals_price.mean()

    return {
        "rmse": np.sqrt(np.mean((preds_price - actuals_price) ** 2)),
        "mae": np.mean(np.abs(preds_price - actuals_price)),
        "mean_price": mean_price,
        "rmse_pct": 100 * np.sqrt(np.mean((preds_price - actuals_price) ** 2)) / mean_price,
        "preds_price": preds_price,
        "actuals_price": actuals_price,
    }