"""Generate report.md summarising all experiment results."""
import json
import os


def generate_report(results: list[dict], output_dir: str) -> None:
    """
    Write report.md to output_dir.

    Each entry in results must have "model" and "dataset" keys, plus either
    metric keys (accuracy, f1_macro, f1_weighted, precision_macro, recall_macro)
    or an "error" key with an error message string.
    """
    datasets = sorted({r["dataset"] for r in results})
    models   = sorted({r["model"]   for r in results})

    idx: dict[tuple, dict] = {(r["model"], r["dataset"]): r for r in results}
    n_folds = next((r["n_folds"] for r in results if r.get("n_folds")), None)

    lines = [
        "# Emotion Recognition — Results Summary",
        "",
        f"Experiments: {len(models)} models × {len(datasets)} datasets",
    ]
    if n_folds:
        lines.append(
            f"F1 (macro) is reported as mean ± 95% CI across {n_folds}-fold stratified cross-validation."
        )
    lines.append("")

    for dataset in datasets:
        lines += [
            f"## Dataset: `{dataset}`",
            "",
            "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | F1 (weighted) |",
            "|-------|:--------:|:-----------------:|:--------------:|:--------------------:|:-------------:|",
        ]
        for model in models:
            r = idx.get((model, dataset), {})
            if not r:
                lines.append(f"| `{model}` | — | — | — | — | — |")
            elif "error" in r:
                short_err = r["error"][:80].replace("|", "\\|")
                lines.append(f"| `{model}` | ❌ `{short_err}` | — | — | — | — |")
            else:
                f1_macro  = r.get("f1_macro", 0)
                f1_margin = r.get("f1_macro_ci95_margin")
                f1_cell = f"{f1_macro:.4f} ± {f1_margin:.4f}" if f1_margin is not None else f"{f1_macro:.4f}"
                lines.append(
                    f"| `{model}` "
                    f"| {r.get('accuracy', 0):.4f} "
                    f"| {r.get('precision_macro', 0):.4f} "
                    f"| {r.get('recall_macro', 0):.4f} "
                    f"| {f1_cell} "
                    f"| {r.get('f1_weighted', 0):.4f} |"
                )
        lines.append("")

    # Per-language F1-macro breakdown (mean across CV folds), for datasets
    # that produced cv_metrics_by_language.json (e.g. multilingual CAMEO).
    for dataset in datasets:
        lang_f1: dict[str, dict[str, float]] = {}
        for model in models:
            lang_path = os.path.join(output_dir, f"{model}_{dataset}", "cv_metrics_by_language.json")
            if not os.path.exists(lang_path):
                continue
            with open(lang_path) as f:
                lang_metrics = json.load(f)
            for lang, m in lang_metrics.items():
                lang_f1.setdefault(lang, {})[model] = m["f1_macro"]["mean"]

        if lang_f1:
            lines += [
                f"## Dataset: `{dataset}` — Language Breakdown (F1-Macro, mean across folds)",
                "",
                "| Language | " + " | ".join(f"`{m}`" for m in models) + " |",
                "|----------|" + "|".join([":------:"] * len(models)) + "|",
            ]
            for lang in sorted(lang_f1):
                row = f"| {lang} "
                for model in models:
                    v = lang_f1[lang].get(model)
                    row += f"| {v:.4f} " if v is not None else "| — "
                row += "|"
                lines.append(row)
            lines.append("")

    if len(datasets) > 1:
        lines += [
            "## Cross-Dataset Comparison — F1-Macro [95% CI]",
            "",
            "| Model | " + " | ".join(f"`{d}`" for d in datasets) + " |",
            "|-------|" + "|".join([":------:"] * len(datasets)) + "|",
        ]
        for model in models:
            row = f"| `{model}` "
            for d in datasets:
                r = idx.get((model, d), {})
                if not r:
                    row += "| — "
                elif "error" in r:
                    row += "| ❌ "
                else:
                    f1_macro  = r.get("f1_macro", 0)
                    f1_margin = r.get("f1_macro_ci95_margin")
                    cell = f"{f1_macro:.4f} ± {f1_margin:.4f}" if f1_margin is not None else f"{f1_macro:.4f}"
                    row += f"| {cell} "
            row += "|"
            lines.append(row)
        lines.append("")

    report = "\n".join(lines)
    path = os.path.join(output_dir, "report.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"\nReport saved -> {path}")
    print(report)
