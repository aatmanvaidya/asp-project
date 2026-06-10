import os
import sys

# ── FFmpeg / torchcodec workaround ────────────────────────────────────────────
# This HPC cluster has no FFmpeg, so torchcodec cannot be used. We must force
# the soundfile backend at every layer before anything else is imported.

# 1. Environment variable picked up by datasets at import time and by any
#    subprocess / worker that inherits the environment.
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# 2. Stub out torchcodec so that `import torchcodec` raises ImportError
#    everywhere (parent process and forked workers).
sys.modules["torchcodec"] = None  # type: ignore[assignment]

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# 3. Patch datasets config flags before the datasets package finishes loading.
import datasets.config
datasets.config.TORCHCODEC_AVAILABLE = False
datasets.config.SOUNDFILE_AVAILABLE = True

from datasets import Audio, load_dataset

# 4. Monkey-patch Audio.decode_example so that even inside .map() worker
#    processes (which re-import datasets fresh) the torchcodec path is never
#    reached. We replace the method with one that always uses soundfile/librosa.
import soundfile as _sf
import numpy as _np

def _decode_example_soundfile(self, value, token_per_repo_id=None):
    """Decode audio using soundfile, bypassing torchcodec entirely."""
    if value is None:
        return value
    # value is already decoded (dict with 'array' and 'sampling_rate')
    if isinstance(value, dict) and "array" in value:
        return value
    # value is a raw bytes / path dict from the Arrow table
    path = value.get("path") or ""
    bytes_ = value.get("bytes")
    if bytes_:
        import io
        array, sampling_rate = _sf.read(io.BytesIO(bytes_), dtype="float32", always_2d=False)
    elif path:
        array, sampling_rate = _sf.read(path, dtype="float32", always_2d=False)
    else:
        return value
    # Resample if needed
    target_sr = self.sampling_rate
    if target_sr is not None and sampling_rate != target_sr:
        import librosa
        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=target_sr)
        sampling_rate = target_sr
    return {"path": path, "array": array.astype(_np.float32), "sampling_rate": sampling_rate}

Audio.decode_example = _decode_example_soundfile
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

warnings.filterwarnings("ignore")


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "facebook/wav2vec2-base"
DATASET_NAME = "xbgoose/ravdess"

SAMPLING_RATE = 16_000
MAX_DURATION = 5.0
MAX_LENGTH = int(SAMPLING_RATE * MAX_DURATION)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
OUTPUT_DIR = "./wav2vec2-ravdess-output"
SEED = 42

RAVDESS_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Wav2Vec2 for RAVDESS emotion classification"
    )
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max_duration", type=float, default=MAX_DURATION)
    return parser.parse_args()


def load_and_explore(dataset_name):
    print("\n[1/7] Loading dataset ...")
    raw_dataset = load_dataset(dataset_name)
    print(raw_dataset)

    base_split = list(raw_dataset.keys())[0]

    # Access column names via the features schema — avoids triggering audio
    # decoding (which would require FFmpeg/torchcodec) at exploration time.
    features = raw_dataset[base_split].features
    example_keys = list(features.keys())

    print("Available splits :", list(raw_dataset.keys()))
    print("Columns          :", example_keys)

    label_col = next(
        (c for c in ["label", "emotion", "labels", "target"] if c in features),
        None,
    )
    assert label_col is not None, "Could not detect label column."
    print(f'Label column     : "{label_col}"')

    # Read only the label column from the Arrow table — no audio decode needed.
    all_labels = raw_dataset[base_split].with_format("numpy")[label_col]

    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"Unique labels : {unique}")
    print(f"Total samples : {len(all_labels):,}")
    for u, c in zip(unique, counts):
        print(f"  {u:>12} -> {c} samples")

    return raw_dataset, base_split, label_col, all_labels


def build_label_maps(raw_dataset, base_split, label_col, all_labels):
    print("\n[2/7] Building label maps ...")
    features = raw_dataset[base_split].features

    # Detect whether labels are strings or integers
    labels_are_strings = isinstance(all_labels[0], (str, np.str_))

    if hasattr(features.get(label_col), "names"):
        # datasets ClassLabel: already has a canonical name list
        label_names = features[label_col].names
        print("Using dataset ClassLabel names:", label_names)
    elif labels_are_strings:
        # String labels (e.g. 'angry', 'calm', ...): sort alphabetically for
        # a stable, reproducible mapping, then build id↔label dicts.
        label_names = sorted(set(str(l) for l in all_labels))
        print("Detected string labels. Sorted label names:", label_names)
    else:
        # Integer labels: use the pre-defined RAVDESS name list
        min_label = int(min(all_labels))
        n_labels = int(max(all_labels)) - min_label + 1
        label_names = RAVDESS_NAMES[:n_labels]
        if min_label != 0:
            print(
                f"Labels start at {min_label}; will shift to 0-indexed during preprocessing."
            )
        print("Using default RAVDESS names:", label_names)

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    num_labels = len(label_names)
    print(f"id2label   : {id2label}")
    print(f"Num classes: {num_labels}")
    return label_names, id2label, label2id, num_labels


def split_dataset(raw_dataset, base_split, label_col, all_labels, seed):
    print("\n[3/7] Splitting dataset (70/15/15) ...")
    base_data = raw_dataset[base_split]
    indices = list(range(len(base_data)))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=all_labels,
        random_state=seed,
    )
    temp_labels = [all_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_labels,
        random_state=seed,
    )

    train_raw = base_data.select(train_idx)
    val_raw = base_data.select(val_idx)
    test_raw = base_data.select(test_idx)

    n = len(base_data)
    print(f"Train : {len(train_raw):>5,} samples  ({len(train_raw) / n * 100:.1f}%)")
    print(f"Val   : {len(val_raw):>5,} samples  ({len(val_raw) / n * 100:.1f}%)")
    print(f"Test  : {len(test_raw):>5,} samples  ({len(test_raw) / n * 100:.1f}%)")
    return train_raw, val_raw, test_raw


def preprocess_splits(
    train_raw, val_raw, test_raw, label_col, all_labels, label2id, model_name, max_length
):
    print("\n[4/7] Extracting features ...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    print("Feature extractor sampling rate:", feature_extractor.sampling_rate)

    # Cast to Audio with the target sampling rate; soundfile will handle decoding
    # since DATASETS_AUDIO_BACKEND=soundfile was set at process start.
    train_raw = train_raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    val_raw = val_raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    test_raw = test_raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    labels_are_strings = isinstance(all_labels[0], (str, np.str_))
    _min_label = 0 if labels_are_strings else int(min(all_labels))

    def preprocess(batch):
        # Process each sample individually to avoid the feature extractor's
        # internal batched tensor-stacking logic, which fails on some
        # transformers versions even with return_tensors=None.
        all_input_values = []
        all_attention_masks = []

        for x in batch["audio"]:
            arr = np.array(x["array"], dtype=np.float32)
            # Truncate or zero-pad to exactly max_length
            if len(arr) >= max_length:
                arr = arr[:max_length]
            else:
                arr = np.pad(arr, (0, max_length - len(arr)))

            # Single-sample call always produces a homogeneous result
            out = feature_extractor(
                arr,
                sampling_rate=SAMPLING_RATE,
                return_attention_mask=True,
                return_tensors=None,
            )
            # Unwrap any extra nesting added by the feature extractor for
            # single-sample calls, e.g. [[...]] -> [...], so each row in the
            # Arrow table is a flat 1-D sequence of length max_length.
            # Without this, set_format("torch") produces [1, 80000] tensors
            # which conv1d rejects (it expects [batch, 1, 80000]).
            iv = np.array(out["input_values"], dtype=np.float32).squeeze()
            am = np.array(out["attention_mask"], dtype=np.int64).squeeze()
            all_input_values.append(iv.tolist())
            all_attention_masks.append(am.tolist())

        if labels_are_strings:
            labels = [label2id[str(l)] for l in batch[label_col]]
        else:
            labels = [int(l) - _min_label for l in batch[label_col]]

        return {
            "input_values": all_input_values,
            "attention_mask": all_attention_masks,
            "labels": labels,
        }

    cols_to_remove = train_raw.column_names
    train_ds = train_raw.map(
        preprocess,
        batched=True,
        num_proc=1,
        remove_columns=cols_to_remove,
        desc="Preprocessing train",
    )
    val_ds = val_raw.map(
        preprocess,
        batched=True,
        num_proc=1,
        remove_columns=cols_to_remove,
        desc="Preprocessing val  ",
    )
    test_ds = test_raw.map(
        preprocess,
        batched=True,
        num_proc=1,
        remove_columns=cols_to_remove,
        desc="Preprocessing test ",
    )

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    print("Columns after preprocessing:", train_ds.column_names)
    print("input_values shape (first sample):", train_ds[0]["input_values"].shape)
    print("Sample label:", train_ds[0]["labels"].item())
    return train_ds, val_ds, test_ds


def load_model(model_name, num_labels, id2label, label2id):
    print("\n[5/7] Loading model ...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "accuracy": float((preds == labels).mean()),
    }


def train(model, train_ds, val_ds, args):
    print("\n[6/7] Training ...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    print("\nTraining complete.")
    print(f"Runtime  : {train_result.metrics['train_runtime']:.1f}s")
    print(f"Samples/s: {train_result.metrics['train_samples_per_second']:.2f}")
    return trainer


def evaluate_test(trainer, test_ds, id2label, num_labels, output_dir):
    print("\n[7/7] Evaluating on test set ...")
    test_output = trainer.predict(test_ds)
    test_preds = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids
    label_names = [id2label[i] for i in range(num_labels)]

    print(f"Test samples: {len(test_labels):,}")
    print("=" * 64)
    print("CLASSIFICATION REPORT")
    print("=" * 64)
    print(
        classification_report(
            test_labels, test_preds, target_names=label_names, zero_division=0
        )
    )

    f1_macro = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
    accuracy = float((test_preds == test_labels).mean())
    print(f"F1-Macro    : {f1_macro:.4f}")
    print(f"F1-Weighted : {f1_weighted:.4f}")
    print(f"Accuracy    : {accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(
        f"Confusion Matrix — Test Set  (F1-Macro={f1_macro:.3f})", fontsize=14, pad=14
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved -> {cm_path}")

    # Training curves
    log_history = trainer.state.log_history
    train_steps = [
        e["step"] for e in log_history if "loss" in e and "eval_loss" not in e
    ]
    train_losses = [
        e["loss"] for e in log_history if "loss" in e and "eval_loss" not in e
    ]
    eval_epochs = [e["epoch"] for e in log_history if "eval_loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    eval_f1 = [e.get("eval_f1_macro") for e in log_history if "eval_loss" in e]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        train_steps, train_losses, color="steelblue", alpha=0.85, linewidth=1.2
    )
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    ax_loss = axes[1]
    ax_loss.plot(eval_epochs, eval_losses, color="coral", marker="o", label="Val Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", color="coral")
    ax_loss.tick_params(axis="y", labelcolor="coral")
    ax_loss.set_title("Validation Loss & F1-Macro per Epoch")

    if any(v is not None for v in eval_f1):
        ax_f1 = ax_loss.twinx()
        ax_f1.plot(
            eval_epochs,
            eval_f1,
            color="seagreen",
            marker="s",
            linestyle="--",
            label="Val F1-Macro",
        )
        ax_f1.set_ylabel("F1-Macro", color="seagreen")
        ax_f1.tick_params(axis="y", labelcolor="seagreen")
        lines = ax_loss.get_lines() + ax_f1.get_lines()
        labels = [l.get_label() for l in lines]
        ax_loss.legend(lines, labels, loc="upper right")
    else:
        ax_loss.legend(loc="upper right")

    ax_loss.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved -> {curves_path}")


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device    : {device}")
    print(f"Model     : {args.model_name}")
    print(f"Dataset   : {args.dataset_name}")
    print(f"Epochs    : {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LR        : {args.learning_rate}")
    max_length = int(SAMPLING_RATE * args.max_duration)
    print(
        f"Max length: {max_length:,} samples ({args.max_duration}s @ {SAMPLING_RATE} Hz)"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    raw_dataset, base_split, label_col, all_labels = load_and_explore(args.dataset_name)
    label_names, id2label, label2id, num_labels = build_label_maps(
        raw_dataset, base_split, label_col, all_labels
    )
    train_raw, val_raw, test_raw = split_dataset(
        raw_dataset, base_split, label_col, all_labels, args.seed
    )
    train_ds, val_ds, test_ds = preprocess_splits(
        train_raw, val_raw, test_raw, label_col, all_labels, label2id, args.model_name, max_length
    )
    model = load_model(args.model_name, num_labels, id2label, label2id)
    trainer = train(model, train_ds, val_ds, args)
    evaluate_test(trainer, test_ds, id2label, num_labels, args.output_dir)


if __name__ == "__main__":
    main()