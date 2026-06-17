"""Entry point for the MFCC + 1D CNN baseline.

Data loaders in data/ should call run(X, y, label_names) directly,
or save features to a .npz and pass the path as a CLI argument:

    uv run python baselines/train.py path/to/features.npz
"""

import csv
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
    train_losses = []
    val_losses, val_f1_macros, val_f1_weighteds, val_accuracies = [], [], [], []
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
        epoch_loss = running_loss / len(X_train)
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        epoch_preds, epoch_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_running_loss += criterion(logits, yb).item() * len(xb)
                epoch_preds.extend(logits.argmax(dim=1).cpu().numpy())
                epoch_true.extend(yb.cpu().numpy())
        val_loss = val_running_loss / len(X_test)
        ep = np.array(epoch_preds)
        et = np.array(epoch_true)
        val_losses.append(val_loss)
        val_f1_macros.append(f1_score(et, ep, average="macro", zero_division=0))
        val_f1_weighteds.append(f1_score(et, ep, average="weighted", zero_division=0))
        val_accuracies.append(float((ep == et).mean()))

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss: {epoch_loss:.4f}  val_loss: {val_loss:.4f}  val_f1: {val_f1_macros[-1]:.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    f1_macro = f1_score(all_true, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_true, all_preds, average="weighted", zero_division=0)
    accuracy = float((all_preds == all_true).mean())

    report = classification_report(all_true, all_preds, target_names=label_names, zero_division=0)
    print("\n=== Classification Report ===")
    print(report)
    print(f"F1-Macro    : {f1_macro:.4f}")
    print(f"F1-Weighted : {f1_weighted:.4f}")
    print(f"Accuracy    : {accuracy:.4f}")

    report_path = os.path.join(RESULTS_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
        f.write(f"\nF1-Macro    : {f1_macro:.4f}\n")
        f.write(f"F1-Weighted : {f1_weighted:.4f}\n")
        f.write(f"Accuracy    : {accuracy:.4f}\n")
    print(f"Saved: {report_path}")

    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — MFCC 1D CNN Baseline  (F1-Macro={f1_macro:.3f})")
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")

    preds_csv_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    with open(preds_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "ground_truth", "predicted", "correct"])
        writer.writeheader()
        for i, (gt, pred) in enumerate(zip(all_true, all_preds)):
            writer.writerow({
                "index": i,
                "ground_truth": label_names[int(gt)],
                "predicted": label_names[int(pred)],
                "correct": int(gt) == int(pred),
            })
    print(f"Saved: {preds_csv_path}")

    loss_csv_path = os.path.join(RESULTS_DIR, "train_loss_history.csv")
    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        for i, loss in enumerate(train_losses, 1):
            writer.writerow({"epoch": i, "train_loss": loss})
    print(f"Saved: {loss_csv_path}")

    val_csv_path = os.path.join(RESULTS_DIR, "val_metrics_history.csv")
    with open(val_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_loss", "f1_macro", "f1_weighted", "accuracy"])
        writer.writeheader()
        for i, (vl, f1m, f1w, acc) in enumerate(zip(val_losses, val_f1_macros, val_f1_weighteds, val_accuracies), 1):
            writer.writerow({"epoch": i, "val_loss": vl, "f1_macro": f1m, "f1_weighted": f1w, "accuracy": acc})
    print(f"Saved: {val_csv_path}")

    epochs_range = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_range, train_losses, color="steelblue", linewidth=1.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    ax_loss = axes[1]
    ax_loss.plot(epochs_range, val_losses, color="coral", marker="o", markersize=3, label="Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color="coral")
    ax_loss.tick_params(axis="y", labelcolor="coral")
    ax_loss.set_title("Validation Loss & F1-Macro per Epoch")
    ax_f1 = ax_loss.twinx()
    ax_f1.plot(epochs_range, val_f1_macros, color="seagreen", marker="s", markersize=3, linestyle="--", label="Val F1-Macro")
    ax_f1.set_ylabel("F1-Macro", color="seagreen")
    ax_f1.tick_params(axis="y", labelcolor="seagreen")
    lines = ax_loss.get_lines() + ax_f1.get_lines()
    ax_loss.legend(lines, [l.get_label() for l in lines], loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved: {curves_path}")

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
