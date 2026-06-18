import os
import sys

# ── FFmpeg / torchcodec workaround ────────────────────────────────────────────
# This HPC cluster has no FFmpeg, so torchcodec cannot be used. We must force
# the soundfile backend at every layer before anything else is imported.
#
# NOTE: The import order here is intentional and load-order-sensitive.
# os/sys must come first so we can stub torchcodec before datasets loads.
# sklearn/transformers imports are deferred below the datasets patch block
# because datasets.config must be patched before datasets submodules load.

# 1. Environment variable picked up by datasets at import time and by any
#    subprocess / worker that inherits the environment.
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# 2. Stub out torchcodec so that `import torchcodec` raises ImportError
#    everywhere (parent process and forked workers).
sys.modules["torchcodec"] = None  # type: ignore[assignment]

import csv
import io
import json
import warnings  # noqa: E402

# 3. Patch datasets config flags before the datasets package finishes loading.
import datasets.config
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import torch

datasets.config.TORCHCODEC_AVAILABLE = False
datasets.config.SOUNDFILE_AVAILABLE = True

from datasets import Audio, load_dataset

# 4. Monkey-patch Audio.decode_example so that even inside .map() worker
#    processes (which re-import datasets fresh) the torchcodec path is never
#    reached. We replace the method with one that always uses soundfile/librosa.


def _decode_example_soundfile(self, value, token_per_repo_id=None):
    """Decode audio using soundfile, bypassing torchcodec entirely."""
    if value is None:
        return value
    # value is already decoded (dict with 'array' and 'sampling_rate')
    if isinstance(value, dict) and "array" in value:
        return value
    # value is a raw bytes / path dict from the Arrow table
    path = value.get("path")
    bytes_ = value.get("bytes")
    if bytes_:
        array, sampling_rate = sf.read(
            io.BytesIO(bytes_), dtype="float32", always_2d=False
        )
    elif path:
        array, sampling_rate = sf.read(path, dtype="float32", always_2d=False)
    else:
        return value
    # Resample if needed
    target_sr = self.sampling_rate
    if target_sr is not None and sampling_rate != target_sr:
        import librosa

        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=target_sr)
        sampling_rate = target_sr
    return {
        "path": path or "",
        "array": array.astype(np.float32),
        "sampling_rate": sampling_rate,
    }


Audio.decode_example = _decode_example_soundfile
# ─────────────────────────────────────────────────────────────────────────────

import functools

import optuna
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

warnings.filterwarnings("ignore")


# ── Configuration ─────────────────────────────────────────────────────────────
# Edit these constants directly — no CLI arguments needed.

MODEL_NAME = "facebook/wav2vec2-base"
DATASET_NAME = "xbgoose/ravdess"
OUTPUT_DIR = "./wav2vec2-ravdess-output"
SEED = 42

# Audio
SAMPLING_RATE = 16_000
MAX_DURATION = 5.0
MAX_LENGTH = int(SAMPLING_RATE * MAX_DURATION)

# Dataset split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Default / fallback hyperparameters (used when HPO is disabled) ──────────
BATCH_SIZE = 8
NUM_EPOCHS = 50  # high ceiling — early stopping will decide when to stop
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# ── Early stopping ──────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 3  # stop after this many epochs without f1 improvement
EARLY_STOPPING_THRESHOLD = 0.001  # minimum improvement to count as progress

# ── Optuna HPO ──────────────────────────────────────────────────────────────
HPO_TRIALS = 20  # set to 0 to skip HPO and use the defaults above
HPO_EPOCHS = 3  # epochs per trial (short — just enough to rank configs)

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

DATASET_HF_MAP: dict[str, str] = {
    "ravdess": "xbgoose/ravdess",
}


# ── Preprocessing helpers ────────────────────────────────────────────────────


def preprocess_batch(
    batch,
    feature_extractor,
    label_col,
    label2id,
    labels_are_strings,
    min_label,
    max_length,
):
    """
    Featurise a batch of audio samples and convert labels to 0-indexed ints.

    Extracted from preprocess_splits() to avoid a nested function definition
    and to make it independently testable.
    """
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
        labels = [label2id[str(l)] for l in batch[label_col]]  # noqa: E741
    else:
        labels = [int(l) - min_label for l in batch[label_col]]  # noqa: E741

    return {
        "input_values": all_input_values,
        "attention_mask": all_attention_masks,
        "labels": labels,
    }


def apply_preprocessing(dataset, preprocess_fn, cols_to_remove, desc):
    """Apply preprocessing map to a dataset split and set torch format."""
    ds = dataset.map(
        preprocess_fn,
        batched=True,
        num_proc=1,
        remove_columns=cols_to_remove,
        desc=desc,
    )
    ds.set_format("torch")
    return ds


# ── Pipeline stages ──────────────────────────────────────────────────────────


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
        label_names = sorted(set(str(l) for l in all_labels))  # noqa: E741
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
    train_raw,
    val_raw,
    test_raw,
    label_col,
    all_labels,
    label2id,
    model_name,
    max_length,
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
    min_label = 0 if labels_are_strings else int(min(all_labels))

    # Build a partial so preprocess_batch can be called with just (batch,)
    # as required by datasets.map(), without needing a nested function.
    preprocess_fn = functools.partial(
        preprocess_batch,
        feature_extractor=feature_extractor,
        label_col=label_col,
        label2id=label2id,
        labels_are_strings=labels_are_strings,
        min_label=min_label,
        max_length=max_length,
    )

    cols_to_remove = train_raw.column_names
    train_ds = apply_preprocessing(
        train_raw, preprocess_fn, cols_to_remove, "Preprocessing train"
    )
    val_ds = apply_preprocessing(
        val_raw, preprocess_fn, cols_to_remove, "Preprocessing val  "
    )
    test_ds = apply_preprocessing(
        test_raw, preprocess_fn, cols_to_remove, "Preprocessing test "
    )

    print("Columns after preprocessing:", train_ds.column_names)
    print("input_values shape (first sample):", train_ds[0]["input_values"].shape)
    print("Sample label:", train_ds[0]["labels"].item())
    return train_ds, val_ds, test_ds, test_raw


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
        "precision_macro": precision_score(
            labels, preds, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "accuracy": float((preds == labels).mean()),
    }


# ── Optuna HPO ───────────────────────────────────────────────────────────────


def run_hpo(train_ds, val_ds, model_name, num_labels, id2label, label2id, output_dir=OUTPUT_DIR):
    """
    Run Optuna hyperparameter search via Trainer.hyperparameter_search().

    Search space
    ------------
    - learning_rate              : log-uniform  [1e-5, 5e-4]
    - per_device_train_batch_size: categorical  [4, 8, 16]
    - warmup_ratio               : uniform      [0.0, 0.2]
    - weight_decay               : uniform      [0.0, 0.1]

    Each trial trains for HPO_EPOCHS epochs. Objective: eval_f1_macro (↑).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    hpo_dir = os.path.join(output_dir, "hpo_trials")
    os.makedirs(hpo_dir, exist_ok=True)

    def model_init(trial=None):
        return Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size", [4, 8, 16]
            ),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        }

    hpo_training_args = TrainingArguments(
        output_dir=hpo_dir,
        num_train_epochs=HPO_EPOCHS,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=500,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model_init=model_init,
        args=hpo_training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\nRunning Optuna HPO: {HPO_TRIALS} trials x {HPO_EPOCHS} epochs each ...")
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=HPO_TRIALS,
        study_name="wav2vec2_ravdess_hpo",
    )

    print("\nBest trial:")
    print(f"  Objective (f1_macro) : {best_run.objective:.4f}")
    print(f"  Hyperparameters      : {best_run.hyperparameters}")

    # Collect all trial results and save to CSV
    csv_path = os.path.join(output_dir, "hpo_trials.csv")
    try:
        study_obj = optuna.load_study(study_name="wav2vec2_ravdess_hpo", storage=None)
        trials = study_obj.trials
        param_keys = sorted({k for t in trials for k in t.params.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["trial_number", "state", "f1_macro"] + param_keys
            )
            writer.writeheader()
            for t in trials:
                row = {
                    "trial_number": t.number,
                    "state": t.state.name,
                    "f1_macro": t.value if t.value is not None else "",
                }
                for k in param_keys:
                    row[k] = t.params.get(k, "")
                writer.writerow(row)
        print(f"Saved HPO trial history ({len(trials)} trials) -> {csv_path}")
    except Exception as e:
        param_keys = sorted(best_run.hyperparameters.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trial", "f1_macro"] + param_keys)
            writer.writeheader()
            writer.writerow(
                {
                    "trial": best_run.run_id,
                    "f1_macro": best_run.objective,
                    **{k: best_run.hyperparameters[k] for k in param_keys},
                }
            )
        print(f"Saved HPO results (best trial only) -> {csv_path}  [{e}]")

    best_params = {
        "learning_rate": best_run.hyperparameters.get("learning_rate", LEARNING_RATE),
        "per_device_train_batch_size": best_run.hyperparameters.get(
            "per_device_train_batch_size", BATCH_SIZE
        ),
        "warmup_ratio": best_run.hyperparameters.get("warmup_ratio", WARMUP_RATIO),
        "weight_decay": best_run.hyperparameters.get("weight_decay", WEIGHT_DECAY),
        "num_epochs": NUM_EPOCHS,
        "hpo_objective_f1_macro": best_run.objective,
        "hpo_trials": HPO_TRIALS,
        "hpo_epochs_per_trial": HPO_EPOCHS,
    }
    json_path = os.path.join(output_dir, "best_hyperparameters.json")
    with open(json_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best hyperparameters -> {json_path}")

    return best_params


def train(
    train_ds, val_ds, model_name, num_labels, id2label, label2id, best_params=None,
    output_dir=OUTPUT_DIR, epochs=NUM_EPOCHS
):
    """
    Full training run with early stopping.
    If best_params is provided (from HPO) those values override the constants above.
    Early stopping monitors eval_f1_macro and stops after EARLY_STOPPING_PATIENCE
    epochs without improvement of at least EARLY_STOPPING_THRESHOLD.

    Checkpointing: save_strategy="best" writes a checkpoint only when
    eval_f1_macro improves; save_total_limit=1 keeps only that single
    best checkpoint on disk at any time.
    """
    print("\n[6/7] Training ...")

    lr = (best_params or {}).get("learning_rate", LEARNING_RATE)
    batch_size = (best_params or {}).get("per_device_train_batch_size", BATCH_SIZE)
    warmup_ratio = (best_params or {}).get("warmup_ratio", WARMUP_RATIO)
    weight_decay = (best_params or {}).get("weight_decay", WEIGHT_DECAY)

    print(f"  learning_rate              : {lr}")
    print(f"  per_device_train_batch_size: {batch_size}")
    print(f"  warmup_ratio               : {warmup_ratio}")
    print(f"  weight_decay               : {weight_decay}")
    print(f"  num_epochs (max)           : {epochs}")
    print(f"  early_stopping_patience    : {EARLY_STOPPING_PATIENCE}")

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="best",  # only save when eval_f1_macro improves
        save_total_limit=1,  # keep exactly one checkpoint on disk at all times
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        seed=SEED,
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    train_result = trainer.train()
    stopped_epoch = int(trainer.state.epoch)
    print("\nTraining complete.")
    print(f"Stopped at epoch : {stopped_epoch} / {NUM_EPOCHS}")
    print(f"Runtime          : {train_result.metrics['train_runtime']:.1f}s")
    print(f"Samples/s        : {train_result.metrics['train_samples_per_second']:.2f}")
    return trainer


# ── Post-training output helpers ─────────────────────────────────────────────


def save_predictions_csv(test_labels, test_preds, test_raw, id2label, output_dir):
    """Write per-sample ground-truth vs prediction to CSV."""
    audio_paths = []
    for sample in test_raw:
        path = ""
        if "audio" in sample and isinstance(sample["audio"], dict):
            path = sample["audio"].get("path") or ""
        audio_paths.append(os.path.basename(path) if path else "")

    predictions_csv = os.path.join(output_dir, "test_predictions.csv")
    with open(predictions_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "audio_file", "ground_truth", "predicted", "correct"],
        )
        writer.writeheader()
        for i, (gt, pred) in enumerate(zip(test_labels, test_preds)):
            writer.writerow(
                {
                    "index": i,
                    "audio_file": audio_paths[i] if i < len(audio_paths) else "",
                    "ground_truth": id2label[int(gt)],
                    "predicted": id2label[int(pred)],
                    "correct": int(gt) == int(pred),
                }
            )
    print(f"Saved -> {predictions_csv}")


def save_confusion_matrix(test_labels, test_preds, label_names, f1_macro, output_dir):
    """Save the confusion matrix heatmap as a PNG."""
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


def save_training_curves(trainer, output_dir):
    """
    Persist training loss history and validation metrics to CSV,
    then plot training loss and validation loss/F1-Macro curves.
    """
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
    eval_f1w = [e.get("eval_f1_weighted") for e in log_history if "eval_loss" in e]
    eval_acc = [e.get("eval_accuracy") for e in log_history if "eval_loss" in e]

    # Per-step training loss CSV
    train_csv_path = os.path.join(output_dir, "train_loss_history.csv")
    with open(train_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss"])
        writer.writeheader()
        for step, loss in zip(train_steps, train_losses):
            writer.writerow({"step": step, "train_loss": loss})
    print(f"Saved -> {train_csv_path}")

    # Per-epoch validation metrics CSV
    val_csv_path = os.path.join(output_dir, "val_metrics_history.csv")
    with open(val_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "val_loss", "f1_macro", "f1_weighted", "accuracy"]
        )
        writer.writeheader()
        for ep, vl, f1, f1w, acc in zip(
            eval_epochs, eval_losses, eval_f1, eval_f1w, eval_acc
        ):
            writer.writerow(
                {
                    "epoch": ep,
                    "val_loss": vl,
                    "f1_macro": f1,
                    "f1_weighted": f1w,
                    "accuracy": acc,
                }
            )
    print(f"Saved -> {val_csv_path}")

    # Training curves plot
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
        labels = [l.get_label() for l in lines]  # noqa: E741
        ax_loss.legend(lines, labels, loc="upper right")
    else:
        ax_loss.legend(loc="upper right")

    ax_loss.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved -> {curves_path}")


def evaluate_test(trainer, test_ds, test_raw, id2label, num_labels, output_dir):
    """Run inference on the test set and print the classification report."""
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

    save_predictions_csv(test_labels, test_preds, test_raw, id2label, output_dir)
    save_confusion_matrix(test_labels, test_preds, label_names, f1_macro, output_dir)
    save_training_curves(trainer, output_dir)


def run(
    dataset_name: str,
    model_name: str = MODEL_NAME,
    output_dir: str = OUTPUT_DIR,
    epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    hpo_trials: int = HPO_TRIALS,
    seed: int = SEED,
) -> None:
    hf_name = DATASET_HF_MAP.get(dataset_name, dataset_name)
    max_length = int(SAMPLING_RATE * MAX_DURATION)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device              : {device}")
    print(f"Model               : {model_name}")
    print(f"Dataset             : {hf_name}")
    print(
        f"Max epochs          : {epochs}  (early stopping patience={EARLY_STOPPING_PATIENCE})"
    )
    print(f"Batch size          : {batch_size}  (may be overridden by HPO)")
    print(f"LR                  : {lr}  (may be overridden by HPO)")
    print(f"HPO trials          : {hpo_trials}  (0 = disabled)")
    print(
        f"Max length          : {max_length:,} samples ({MAX_DURATION}s @ {SAMPLING_RATE} Hz)"
    )

    os.makedirs(output_dir, exist_ok=True)

    raw_dataset, base_split, label_col, all_labels = load_and_explore(hf_name)
    label_names, id2label, label2id, num_labels = build_label_maps(
        raw_dataset, base_split, label_col, all_labels
    )
    train_raw, val_raw, test_raw = split_dataset(
        raw_dataset, base_split, label_col, all_labels, seed
    )
    train_ds, val_ds, test_ds, test_raw = preprocess_splits(
        train_raw,
        val_raw,
        test_raw,
        label_col,
        all_labels,
        label2id,
        model_name,
        max_length,
    )

    # ── Optional HPO phase ────────────────────────────────────────────────
    best_params = None
    if hpo_trials > 0:
        best_params = run_hpo(
            train_ds, val_ds, model_name, num_labels, id2label, label2id,
            output_dir=output_dir,
        )
    else:
        print("\nHPO disabled. Using fixed hyperparameters from config.")
        best_params = {
            "learning_rate": lr,
            "per_device_train_batch_size": batch_size,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
        }

    # ── Final training run (with early stopping) ──────────────────────────
    trainer = train(
        train_ds,
        val_ds,
        model_name,
        num_labels,
        id2label,
        label2id,
        best_params=best_params,
        output_dir=output_dir,
        epochs=epochs,
    )
    evaluate_test(trainer, test_ds, test_raw, id2label, num_labels, output_dir)


def main():
    run(
        dataset_name=DATASET_NAME,
        model_name=MODEL_NAME,
        output_dir=OUTPUT_DIR,
        epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        hpo_trials=HPO_TRIALS,
        seed=SEED,
    )


if __name__ == "__main__":
    main()
