"""
Pearson correlation between per-file audio SNR and model correctness.

For each RAVDESS/EmoDB audio file, estimates a blind SNR (see snr.py), then
correlates it against the "correct" (0/1) column pooled from every fold's
test_predictions.csv across the trained model x dataset experiments in
outputs/. Since each experiment's K_FOLDS CV folds partition the whole
dataset, concatenating fold_0..fold_{K-1}/test_predictions.csv for one
experiment covers every audio file in that dataset exactly once (evaluated
from the fold where it was held out).

Only `correct` (0/1) is available per sample -- prediction probabilities
were never persisted and fold checkpoints are deleted after training (see
run_all.py), so this is a point-biserial correlation (Pearson's r with a
binary variable), not a correlation against a continuous confidence score.

Reports two granularities:
  - per model x dataset   -- does SNR sensitivity vary by model?
  - pooled per dataset, across all models -- higher-power headline number
    (note: reuses the same per-file SNR value once per model, so these
    pooled samples are not independent draws -- treat as a descriptive
    pooled effect, not an inflated-N significance test)

Usage:
    python experiments/snr_analysis/run_snr_analysis.py
    python experiments/snr_analysis/run_snr_analysis.py --predictions_root outputs --output_dir outputs/snr_analysis

Outputs (under --output_dir):
    snr_<dataset>.json            per-file SNR cache (audio_file -> snr_db)
    snr_correlation.json          full numeric results
    snr_correlation_report.md     human-readable summary
    snr_vs_accuracy_<dataset>.png accuracy by SNR-quartile, per dataset
"""
import argparse
import csv
import glob
import json
import os
import sys

# ── Audio backend must be patched before ANY datasets/audio import ─────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # experiments/snr_analysis/
import pipeline.audio_backend  # noqa: F401, E402  (side-effect import — must be first)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from pipeline.config import DATASETS, MODELS  # noqa: E402
from snr import compute_dataset_snr  # noqa: E402


def load_experiment_predictions(experiment_dir: str) -> list[dict]:
    """Concatenate audio_file/correct rows across all fold_*/test_predictions.csv."""
    rows = []
    for csv_path in sorted(glob.glob(os.path.join(experiment_dir, "fold_*", "test_predictions.csv"))):
        with open(csv_path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def pearson_point_biserial(snr_values: list[float], correct_values: list[int]) -> dict:
    """Pearson r (point-biserial, since correct is 0/1) between SNR and correctness."""
    snr_arr = np.asarray(snr_values, dtype=float)
    correct_arr = np.asarray(correct_values, dtype=float)
    n = len(snr_arr)

    if n < 2 or np.std(snr_arr) == 0 or np.std(correct_arr) == 0:
        return {"n": n, "r": float("nan"), "p": float("nan"), "accuracy": float(correct_arr.mean()) if n else float("nan")}

    r, p = stats.pearsonr(snr_arr, correct_arr)
    return {"n": n, "r": float(r), "p": float(p), "accuracy": float(correct_arr.mean())}


def snr_quartile_accuracy(snr_values: list[float], correct_values: list[int]) -> dict:
    """Mean accuracy within each SNR quartile (Q1 = noisiest, Q4 = cleanest)."""
    snr_arr = np.asarray(snr_values, dtype=float)
    correct_arr = np.asarray(correct_values, dtype=float)
    edges = np.percentile(snr_arr, [25, 50, 75])
    bin_idx = np.digitize(snr_arr, edges)  # 0..3

    labels = ["Q1 (noisiest)", "Q2", "Q3", "Q4 (cleanest)"]
    result = {}
    for i, label in enumerate(labels):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        result[label] = {"n": int(mask.sum()), "accuracy": float(correct_arr[mask].mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description="Correlate audio SNR with model correctness.")
    parser.add_argument("--predictions_root", default="outputs",
                         help="Root dir containing <model>_<dataset>/fold_*/test_predictions.csv")
    parser.add_argument("--output_dir", default="outputs/snr_analysis")
    parser.add_argument("--force_snr", action="store_true",
                         help="Recompute per-file SNR even if the cache file already exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Per-file SNR, one pass per dataset ──────────────────────────────────
    snr_by_dataset: dict[str, dict[str, float]] = {}
    for dataset_key, hf_dataset in DATASETS.items():
        print(f"\nComputing SNR for {dataset_key} ({hf_dataset})...")
        cache_path = os.path.join(args.output_dir, f"snr_{dataset_key}.json")
        snr_by_dataset[dataset_key] = compute_dataset_snr(hf_dataset, cache_path, force=args.force_snr)

    # ── Join predictions with SNR, per experiment ───────────────────────────
    per_experiment = []
    pooled_by_dataset: dict[str, dict] = {}
    quartiles_by_dataset: dict[str, dict] = {}
    pooled_snr_correct: dict[str, tuple[list[float], list[int]]] = {d: ([], []) for d in DATASETS}

    for dataset_key in DATASETS:
        snr_by_file = snr_by_dataset[dataset_key]

        for model_key in MODELS:
            experiment_dir = os.path.join(args.predictions_root, f"{model_key}_{dataset_key}")
            if not os.path.isdir(experiment_dir):
                print(f"  SKIP {model_key}_{dataset_key}: no output directory")
                continue

            rows = load_experiment_predictions(experiment_dir)
            if not rows:
                print(f"  SKIP {model_key}_{dataset_key}: no test_predictions.csv found")
                continue

            snr_values, correct_values, unmatched = [], [], 0
            for row in rows:
                snr = snr_by_file.get(row["audio_file"])
                if snr is None:
                    unmatched += 1
                    continue
                snr_values.append(snr)
                correct_values.append(1 if row["correct"] == "True" else 0)

            if unmatched:
                print(f"  {model_key}_{dataset_key}: {unmatched} predictions had no matching SNR entry")

            result = pearson_point_biserial(snr_values, correct_values)
            result.update({"model": model_key, "dataset": dataset_key})
            per_experiment.append(result)

            pooled_snr, pooled_correct = pooled_snr_correct[dataset_key]
            pooled_snr.extend(snr_values)
            pooled_correct.extend(correct_values)

        pooled_snr, pooled_correct = pooled_snr_correct[dataset_key]
        if pooled_snr:
            pooled_by_dataset[dataset_key] = pearson_point_biserial(pooled_snr, pooled_correct)
            quartiles_by_dataset[dataset_key] = snr_quartile_accuracy(pooled_snr, pooled_correct)

    # ── SNR distribution per dataset (unique files, not repeated per model) ─
    snr_distribution = {}
    for dataset_key, snr_by_file in snr_by_dataset.items():
        vals = np.array(list(snr_by_file.values()), dtype=float)
        snr_distribution[dataset_key] = {
            "n_files": len(vals),
            "mean":   float(vals.mean()),
            "std":    float(vals.std()),
            "min":    float(vals.min()),
            "max":    float(vals.max()),
            "median": float(np.median(vals)),
        }

    # ── Persist numeric results ─────────────────────────────────────────────
    results = {
        "per_experiment": per_experiment,
        "pooled_by_dataset": pooled_by_dataset,
        "snr_quartile_accuracy_by_dataset": quartiles_by_dataset,
        "snr_distribution": snr_distribution,
    }
    results_path = os.path.join(args.output_dir, "snr_correlation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {results_path}")

    # ── Plots: accuracy by SNR quartile, per dataset ────────────────────────
    for dataset_key, quartiles in quartiles_by_dataset.items():
        labels = list(quartiles.keys())
        accs = [quartiles[label]["accuracy"] for label in labels]
        ns = [quartiles[label]["n"] for label in labels]

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, accs, color="steelblue", alpha=0.85)
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"n={n}", ha="center", fontsize=9)
        ax.set(ylabel="Accuracy (pooled across models)", ylim=(0, 1.05),
               title=f"Accuracy by SNR Quartile — {dataset_key}")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, f"snr_vs_accuracy_{dataset_key}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved -> {plot_path}")

    # ── Markdown report ──────────────────────────────────────────────────
    lines = ["# SNR vs. Model Correctness — Pearson Correlation", ""]
    lines.append(
        "Point-biserial Pearson r between per-file blind SNR (dB) and prediction "
        "correctness (0/1), pooled across all 5 CV folds so every audio file is "
        "covered exactly once per model."
    )
    lines.append("")

    lines += ["## SNR Distribution", "", "| Dataset | Files | Mean | Std | Min | Median | Max |",
              "|---------|:-----:|:----:|:---:|:---:|:------:|:---:|"]
    for dataset_key, d in snr_distribution.items():
        lines.append(
            f"| `{dataset_key}` | {d['n_files']} | {d['mean']:.2f} | {d['std']:.2f} | "
            f"{d['min']:.2f} | {d['median']:.2f} | {d['max']:.2f} |"
        )
    lines.append("")

    lines += ["## Pooled Per-Dataset Correlation (across all models)", "",
              "| Dataset | n | Pearson r | p-value | Accuracy |",
              "|---------|:-:|:---------:|:-------:|:--------:|"]
    for dataset_key, r in pooled_by_dataset.items():
        lines.append(
            f"| `{dataset_key}` | {r['n']} | {r['r']:.4f} | {r['p']:.4g} | {r['accuracy']:.4f} |"
        )
    lines.append("")

    lines += ["## Per Model x Dataset Correlation", "",
              "| Model | Dataset | n | Pearson r | p-value | Accuracy |",
              "|-------|---------|:-:|:---------:|:-------:|:--------:|"]
    for r in sorted(per_experiment, key=lambda x: (x["dataset"], x["model"])):
        lines.append(
            f"| `{r['model']}` | `{r['dataset']}` | {r['n']} | {r['r']:.4f} | "
            f"{r['p']:.4g} | {r['accuracy']:.4f} |"
        )
    lines.append("")

    lines += ["## Accuracy by SNR Quartile (pooled across models)", ""]
    for dataset_key, quartiles in quartiles_by_dataset.items():
        lines.append(f"### `{dataset_key}`")
        lines.append("")
        lines.append("| Quartile | n | Accuracy |")
        lines.append("|----------|:-:|:--------:|")
        for label, q in quartiles.items():
            lines.append(f"| {label} | {q['n']} | {q['accuracy']:.4f} |")
        lines.append("")

    report = "\n".join(lines)
    report_path = os.path.join(args.output_dir, "snr_correlation_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved -> {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
