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

    lines = [
        "# Emotion Recognition — Results Summary",
        "",
        f"Experiments: {len(models)} models × {len(datasets)} datasets",
        "",
    ]

    for dataset in datasets:
        lines += [
            f"## Dataset: `{dataset}`",
            "",
            "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |",
            "|-------|:--------:|:-----------------:|:--------------:|:----------:|:-------------:|",
        ]
        for model in models:
            r = idx.get((model, dataset), {})
            if not r:
                lines.append(f"| `{model}` | — | — | — | — | — |")
            elif "error" in r:
                short_err = r["error"][:80].replace("|", "\\|")
                lines.append(f"| `{model}` | ❌ `{short_err}` | — | — | — | — |")
            else:
                lines.append(
                    f"| `{model}` "
                    f"| {r.get('accuracy', 0):.4f} "
                    f"| {r.get('precision_macro', 0):.4f} "
                    f"| {r.get('recall_macro', 0):.4f} "
                    f"| {r.get('f1_macro', 0):.4f} "
                    f"| {r.get('f1_weighted', 0):.4f} |"
                )
        lines.append("")

    # Per-language F1-macro breakdown, for datasets that produced
    # metrics_by_language.json (e.g. multilingual CAMEO).
    for dataset in datasets:
        lang_f1: dict[str, dict[str, float]] = {}
        for model in models:
            lang_path = os.path.join(output_dir, f"{model}_{dataset}", "metrics_by_language.json")
            if not os.path.exists(lang_path):
                continue
            with open(lang_path) as f:
                lang_metrics = json.load(f)
            for lang, m in lang_metrics.items():
                lang_f1.setdefault(lang, {})[model] = m["f1_macro"]

        if lang_f1:
            lines += [
                f"## Dataset: `{dataset}` — Language Breakdown (F1-Macro)",
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
            "## Cross-Dataset Comparison — F1-Macro",
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
                    row += f"| {r.get('f1_macro', 0):.4f} "
            row += "|"
            lines.append(row)
        lines.append("")

    report = "\n".join(lines)
    path = os.path.join(output_dir, "report.md")
    with open(path, "w") as f:
        f.write(report)
    print(f"\nReport saved -> {path}")
    print(report)
