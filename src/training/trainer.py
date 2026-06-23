import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


def train_pytorch_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 200,
    learning_rate: float = 0.001,
    device: Optional[str] = None,
    early_stopping_patience: int = 30,
) -> tuple[dict, nn.Module]:
    """
    Train a PyTorch model with early stopping.

    At each epoch, train loss and test loss are recorded. If test loss does
    not improve for `early_stopping_patience` consecutive epochs, training
    stops and the best model weights are restored.

    Args:
        model:                    PyTorch model to train.
        train_loader:             DataLoader for training data.
        test_loader:              DataLoader for test data.
        epochs:                   Maximum number of training epochs.
        learning_rate:            Adam optimizer learning rate.
        device:                   Target device. Auto-detected if None
                                  (prefers MPS > CUDA > CPU).
        early_stopping_patience:  Epochs to wait before stopping when
                                  test loss stops improving.

    Returns:
        history:  Dict with 'train_loss' and 'test_loss' lists.
        model:    Model loaded with best weights found during training.
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "test_loss": []}
    best_test_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"Training on: {device}")
    print(f"Epochs: {epochs} | Learning rate: {learning_rate}")
    print("=" * 60)

    start = time.time()

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # -- Evaluate --
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                test_loss += criterion(model(X_batch).squeeze(), y_batch).item()
        test_loss /= len(test_loader)
        history["test_loss"].append(test_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.2f} | Test Loss: {test_loss:.2f}")

        # -- Early stopping --
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break

    print(f"\nTraining completed in {time.time() - start:.2f} seconds")
    return history, model


def save_model(model: nn.Module, save_path: str) -> None:
    """Save model state dict to disk."""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")