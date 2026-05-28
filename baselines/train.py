"""Entry point for the MFCC + 1D CNN baseline.

Data loaders in data/ should call run(X, y, label_names) directly,
or save features to a .npz and pass the path as a CLI argument:

    uv run python baselines/train.py path/to/features.npz
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EmotionCNN

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
RANDOM_STATE = 42
TEST_SIZE = 0.2

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
MODELS_DIR = os.path.join(_HERE, "..", "models")


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    *,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Train and evaluate the 1D CNN baseline.

    Args:
        X: float32 array of shape (N, N_MFCC, T_MAX)
        y: int64 label indices of shape (N,)
        label_names: ordered list of class name strings (index matches y values)
        epochs: number of training epochs
        lr: AdamW learning rate
        batch_size: DataLoader batch size
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    N, C, T = X.shape
    n_classes = len(label_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(-1, C, T).astype(np.float32)
    X_test = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(-1, C, T).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
    )

    device = _get_device()
    print(f"Training on {device}  |  train={len(X_train)}  test={len(X_test)}\n")

    model = EmotionCNN(n_mfcc=C, n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss: {running_loss / len(X_train):.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    report = classification_report(all_true, all_preds, target_names=label_names)
    print("\n=== Classification Report ===")
    print(report)

    report_path = os.path.join(RESULTS_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — MFCC 1D CNN Baseline")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")

    model_path = os.path.join(MODELS_DIR, "mfcc_cnn.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run python baselines/train.py <path/to/features.npz>")
        print("  The .npz must contain: X (N, C, T), y (N,), label_names (list of str)")
        sys.exit(1)

    data = np.load(sys.argv[1], allow_pickle=True)
    run(
        X=data["X"],
        y=data["y"],
        label_names=data["label_names"].tolist(),
    )
