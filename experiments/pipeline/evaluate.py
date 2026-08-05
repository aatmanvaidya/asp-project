"""Post-training evaluation: metrics, CSVs, and plots."""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def run_inference(trainer, test_ds):
    """Run prediction on test_ds; returns PredictionOutput."""
    return trainer.predict(test_ds)


def aggregate_cv_metrics(fold_metrics: list[dict], output_dir: str) -> dict:
    """
    Aggregate per-fold test metrics (each a dict as returned by save_results)
    into a mean and 95% CI (t-distribution, df = n_folds - 1) per metric.

    Persists the full per-fold detail to cv_metrics.json and returns a flat
    dict — {metric, metric_std, metric_ci95_margin, metric_ci95_lower,
    metric_ci95_upper, ...} — suitable for the results list consumed by
    report.generate_report().
    """
    from scipy import stats

    n = len(fold_metrics)
    metric_keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))]
    t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else float("nan")

    detail: dict = {}
    flat: dict = {"n_folds": n}
    for key in metric_keys:
        vals = np.array([m[key] for m in fold_metrics], dtype=float)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        margin = float(t_crit * std / np.sqrt(n)) if n > 1 else 0.0
        detail[key] = {
            "mean": mean,
            "std": std,
            "ci95_margin": margin,
            "ci95_lower": mean - margin,
            "ci95_upper": mean + margin,
            "fold_values": vals.tolist(),
        }
        flat[key] = mean
        flat[f"{key}_std"] = std
        flat[f"{key}_ci95_margin"] = margin
        flat[f"{key}_ci95_lower"] = mean - margin
        flat[f"{key}_ci95_upper"] = mean + margin

    with open(os.path.join(output_dir, "cv_metrics.json"), "w") as f:
        json.dump({"n_folds": n, "metrics": detail}, f, indent=2)

    print(f"\n  {n}-fold CV summary (mean ± 95% CI):")
    for key in ["f1_macro", "f1_weighted", "precision_macro", "recall_macro", "accuracy"]:
        if key in detail:
            d = detail[key]
            print(
                f"    {key:16s} = {d['mean']:.4f} ± {d['ci95_margin']:.4f}"
                f"  (95% CI [{d['ci95_lower']:.4f}, {d['ci95_upper']:.4f}])"
            )

    return flat


def aggregate_cv_language_metrics(fold_lang_metrics: list[dict], output_dir: str) -> dict:
    """
    Aggregate per-fold per-language metrics (each element is one fold's
    metrics_by_language.json content) into mean ± 95% CI per language, across
    the folds where that language appeared in the test fold. Persists to
    cv_metrics_by_language.json. No-op if fold_lang_metrics is empty (i.e.
    dataset has no "language" column, such as RAVDESS/EMODB).
    """
    from scipy import stats

    if not fold_lang_metrics:
        return {}

    languages = sorted({lang for fm in fold_lang_metrics for lang in fm})
    result: dict = {}
    for lang in languages:
        per_fold = [fm[lang] for fm in fold_lang_metrics if lang in fm]
        n = len(per_fold)
        metric_keys = [k for k, v in per_fold[0].items() if isinstance(v, (int, float))]
        t_crit = float(stats.t.ppf(0.975, df=n - 1)) if n > 1 else float("nan")

        lang_summary = {"n_folds": n}
        for key in metric_keys:
            vals = np.array([m[key] for m in per_fold], dtype=float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            margin = float(t_crit * std / np.sqrt(n)) if n > 1 else 0.0
            lang_summary[key] = {
                "mean": mean,
                "std": std,
                "ci95_margin": margin,
                "ci95_lower": mean - margin,
                "ci95_upper": mean + margin,
            }
        result[lang] = lang_summary

    with open(os.path.join(output_dir, "cv_metrics_by_language.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_cv_metrics(output_dir: str) -> dict:
    """Load a previously saved cv_metrics.json, flattened like aggregate_cv_metrics()
    returns — used to repopulate the results/report when --skip_done skips an
    already-completed experiment."""
    with open(os.path.join(output_dir, "cv_metrics.json")) as f:
        cv_data = json.load(f)

    flat = {"n_folds": cv_data.get("n_folds")}
    for key, d in cv_data.get("metrics", {}).items():
        flat[key] = d["mean"]
        flat[f"{key}_std"] = d["std"]
        flat[f"{key}_ci95_margin"] = d["ci95_margin"]
        flat[f"{key}_ci95_lower"] = d["ci95_lower"]
        flat[f"{key}_ci95_upper"] = d["ci95_upper"]
    return flat


def save_results(
    pred_output,
    test_raw,
    id2label: dict,
    num_labels: int,
    output_dir: str,
) -> dict:
    """
    Compute and persist test metrics, predictions CSV, confusion matrix, and
    training curves.
    """
    preds  = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids
    label_names = [id2label[i] for i in range(num_labels)]

    print(classification_report(labels, preds, target_names=label_names, zero_division=0))

    metrics = {
        "accuracy":        float((preds == labels).mean()),
        "f1_macro":        f1_score(labels, preds, average="macro",    zero_division=0),
        "f1_weighted":     f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":    recall_score(labels, preds, average="macro",    zero_division=0),
    }

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    report_dict = classification_report(
        labels, preds, target_names=label_names, zero_division=0, output_dict=True
    )
    with open(os.path.join(output_dir, "classification_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    _save_predictions_csv(labels, preds, test_raw, id2label, output_dir)
    _save_confusion_matrix(labels, preds, label_names, metrics["f1_macro"], output_dir)
    _save_language_breakdown(labels, preds, test_raw, output_dir)

    print(
        f"  F1-Macro {metrics['f1_macro']:.4f} | "
        f"Accuracy {metrics['accuracy']:.4f} | "
        f"Precision {metrics['precision_macro']:.4f} | "
        f"Recall {metrics['recall_macro']:.4f}"
    )
    return metrics


def save_training_curves(trainer, output_dir: str) -> None:
    """Persist training loss and val metrics to CSVs and a combined PNG."""
    log = trainer.state.log_history

    train_steps  = [e["step"]  for e in log if "loss" in e and "eval_loss" not in e]
    train_losses = [e["loss"]  for e in log if "loss" in e and "eval_loss" not in e]
    eval_epochs  = [e["epoch"] for e in log if "eval_loss" in e]
    eval_losses  = [e["eval_loss"] for e in log if "eval_loss" in e]
    eval_f1      = [e.get("eval_f1_macro")    for e in log if "eval_loss" in e]
    eval_f1w     = [e.get("eval_f1_weighted") for e in log if "eval_loss" in e]
    eval_acc     = [e.get("eval_accuracy")    for e in log if "eval_loss" in e]

    with open(os.path.join(output_dir, "train_loss_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss"])
        w.writeheader()
        for s, l in zip(train_steps, train_losses):
            w.writerow({"step": s, "train_loss": l})

    with open(os.path.join(output_dir, "val_metrics_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_loss", "f1_macro", "f1_weighted", "accuracy"])
        w.writeheader()
        for ep, vl, f1, f1w, acc in zip(eval_epochs, eval_losses, eval_f1, eval_f1w, eval_acc):
            w.writerow({"epoch": ep, "val_loss": vl, "f1_macro": f1, "f1_weighted": f1w, "accuracy": acc})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_steps, train_losses, color="steelblue", alpha=0.85, linewidth=1.2)
    axes[0].set(xlabel="Step", ylabel="Cross-Entropy Loss", title="Training Loss")
    axes[0].grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(eval_epochs, eval_losses, color="coral", marker="o", label="Val Loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Validation Loss & F1-Macro")
    ax.tick_params(axis="y", labelcolor="coral")
    if any(v is not None for v in eval_f1):
        ax2 = ax.twinx()
        ax2.plot(eval_epochs, eval_f1, color="seagreen", marker="s", linestyle="--", label="Val F1-Macro")
        ax2.set_ylabel("F1-Macro", color="seagreen")
        ax2.tick_params(axis="y", labelcolor="seagreen")
        lines  = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], loc="upper right")  # noqa: E741
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()


def _save_language_breakdown(labels, preds, test_raw, output_dir):
    """Per-language accuracy/F1 for multilingual datasets (e.g. CAMEO); no-op otherwise."""
    if "language" not in test_raw.column_names:
        return

    languages = test_raw["language"]
    breakdown = {}
    for lang in sorted(set(languages)):
        idxs = [i for i, lg in enumerate(languages) if lg == lang]
        y_true = [labels[i] for i in idxs]
        y_pred = [preds[i]  for i in idxs]
        breakdown[lang] = {
            "n_samples":       len(idxs),
            "accuracy":        float(np.mean([t == p for t, p in zip(y_true, y_pred)])),
            "f1_macro":        f1_score(y_true, y_pred, average="macro",    zero_division=0),
            "f1_weighted":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro":    recall_score(y_true, y_pred, average="macro",    zero_division=0),
        }

    with open(os.path.join(output_dir, "metrics_by_language.json"), "w") as f:
        json.dump(breakdown, f, indent=2)

    print("  Per-language breakdown:")
    for lang, m in breakdown.items():
        print(f"    {lang:12s} n={m['n_samples']:4d}  F1-Macro={m['f1_macro']:.4f}")


def _save_predictions_csv(labels, preds, test_raw, id2label, output_dir):
    paths = []
    for sample in test_raw:
        p = ""
        if isinstance(sample.get("audio"), dict):
            p = os.path.basename(sample["audio"].get("path") or "")
        paths.append(p)

    with open(os.path.join(output_dir, "test_predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["index", "audio_file", "ground_truth", "predicted", "correct"]
        )
        w.writeheader()
        for i, (gt, pr) in enumerate(zip(labels, preds)):
            w.writerow({
                "index":        i,
                "audio_file":   paths[i] if i < len(paths) else "",
                "ground_truth": id2label[int(gt)],
                "predicted":    id2label[int(pr)],
                "correct":      int(gt) == int(pr),
            })


def _save_confusion_matrix(labels, preds, label_names, f1_macro, output_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix  (F1-Macro={f1_macro:.3f})", fontsize=14, pad=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # Row-normalized (per true class) — each row sums to 1, so classes with
    # few test samples are as visible as high-count classes.
    cm_norm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0,
        xticklabels=label_names, yticklabels=label_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — Normalized by True Class  (F1-Macro={f1_macro:.3f})", fontsize=14, pad=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_normalised.png"), dpi=150)
    plt.close()
