"""Training loop, compute_metrics, and optional Optuna HPO."""
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    HPO_EPOCHS,
    LEARNING_RATE,
    NUM_EPOCHS,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro":        f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted":     f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":    recall_score(labels, preds, average="macro",    zero_division=0),
        "accuracy":        float((preds == labels).mean()),
    }


def run_hpo(
    train_ds,
    val_ds,
    hf_name: str,
    num_labels: int,
    id2label: dict,
    label2id: dict,
    output_dir: str,
    n_trials: int,
    epochs: int = HPO_EPOCHS,
) -> dict:
    """
    Optuna hyperparameter search via Trainer.hyperparameter_search().
    Only call from the main process; not compatible with DDP.
    """
    import optuna
    from .model import load_model

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    hpo_dir = os.path.join(output_dir, "hpo_trials")
    os.makedirs(hpo_dir, exist_ok=True)

    def model_init(trial=None):
        return load_model(hf_name, num_labels, id2label, label2id)

    def hp_space(trial):
        return {
            "learning_rate":               trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
            "warmup_ratio":                trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "weight_decay":                trial.suggest_float("weight_decay", 0.0, 0.1),
        }

    hpo_args = TrainingArguments(
        output_dir=hpo_dir,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(
        model_init=model_init,
        args=hpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print(f"  HPO: {n_trials} trials × {epochs} epochs (on {len(train_ds):,} examples) ...")
    best = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=n_trials,
        study_name="hpo",
    )
    print(f"  Best f1_macro: {best.objective:.4f}  |  params: {best.hyperparameters}")

    best_params = {
        "learning_rate":               best.hyperparameters.get("learning_rate", LEARNING_RATE),
        "per_device_train_batch_size": best.hyperparameters.get("per_device_train_batch_size", BATCH_SIZE),
        "warmup_ratio":                best.hyperparameters.get("warmup_ratio", WARMUP_RATIO),
        "weight_decay":                best.hyperparameters.get("weight_decay", WEIGHT_DECAY),
        "hpo_objective_f1_macro":      best.objective,
    }
    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    return best_params


def train_model(
    train_ds,
    val_ds,
    model,
    output_dir: str,
    lr: float       = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    warmup_ratio: float = WARMUP_RATIO,
    weight_decay: float = WEIGHT_DECAY,
    epochs: int     = NUM_EPOCHS,
) -> Trainer:
    """Full training run with early stopping. Returns the fitted Trainer."""
    print(
        f"  Training  : lr={lr}  bs={batch_size}  "
        f"epochs={epochs}  patience={EARLY_STOPPING_PATIENCE}"
    )
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="best",
        save_total_limit=1,
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
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
        )],
    )
    result = trainer.train()
    print(
        f"  Finished  : epoch {int(trainer.state.epoch)}/{epochs} | "
        f"runtime {result.metrics['train_runtime']:.0f}s | "
        f"{result.metrics['train_samples_per_second']:.1f} samples/s"
    )
    return trainer
