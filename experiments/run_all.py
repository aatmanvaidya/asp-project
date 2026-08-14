"""
Emotion recognition pipeline — runs all model × dataset combinations.
Single GPU throughout — no DDP/torchrun.

Two-stage workflow (see run_pipeline.sh):

  Stage 1 — HPO (Optuna finds best hyperparameters per experiment, tuned once
    on a single held-out split — not re-tuned per CV fold):
    python experiments/run_all.py --hpo_trials 20 --hpo_only

  Stage 2 — K_FOLDS-fold CV training (loads Stage-1 hyperparameters,
    trains+evaluates K_FOLDS fresh models, and reports F1-macro etc. as
    mean ± 95% CI across folds):
    python experiments/run_all.py --skip_done

Single-GPU end-to-end (HPO + CV training in one go):
    python experiments/run_all.py --hpo_trials 20

Additional flags:
    --output_dir  outputs/         root directory for all experiment outputs
    --epochs      30               max training epochs per fold (early stopping may stop sooner)
    --batch_size  8
    --lr          3e-5
    --seed        42
    --k_folds     5                number of stratified CV folds in Stage 2
    --models      wav2vec2-base …  subset of models to run (default: all)
    --datasets    ravdess emodb    subset of datasets to run (default: all)
    --skip_done                    skip already-completed folds/experiments (cv_metrics.json / fold_*/metrics.json)
    --hpo_only                     run HPO only, save best_hyperparameters.json, skip training
"""
import argparse
import glob
import json
import os
import shutil
import sys
import traceback

# ── Audio backend must be patched before ANY datasets/audio import ─────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline.audio_backend  # noqa: F401, E402  (side-effect import — must be first)

import torch  # noqa: E402

from pipeline.config import (  # noqa: E402
    BATCH_SIZE,
    CAMEO_CORE_EMOTIONS,
    DATASETS,
    HPO_SUBSAMPLE_FRACTION,
    HPO_SUBSAMPLE_THRESHOLD,
    HPO_TRIALS,
    K_FOLDS,
    LEARNING_RATE,
    MODELS,
    NUM_EPOCHS,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from pipeline.data import (  # noqa: E402
    build_label_maps,
    decode_split,
    featurise_split,
    filter_labels,
    kfold_split_dataset,
    load_dataset_hf,
    split_dataset,
    subsample_for_hpo,
)
from pipeline.evaluate import (  # noqa: E402
    aggregate_cv_language_metrics,
    aggregate_cv_metrics,
    load_cv_metrics,
    run_inference,
    save_results,
    save_training_curves,
)
from pipeline.model import load_feature_extractor, load_model  # noqa: E402
from pipeline.report import generate_report  # noqa: E402
from pipeline.trainer_utils import run_hpo, train_model  # noqa: E402


def run_experiment(
    model_key: str,
    hf_model:  str,
    dataset_key: str,
    hf_dataset: str,
    output_dir: str,
    lr: float       = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    epochs: int     = NUM_EPOCHS,
    hpo_trials: int = HPO_TRIALS,
    k_folds: int    = K_FOLDS,
    seed: int       = SEED,
    hpo_only: bool  = False,
    skip_done: bool = False,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    raw, base_split, label_col, all_labels = load_dataset_hf(hf_dataset)

    # CAMEO spans 13 sub-corpora with only partially overlapping emotion
    # vocabularies — restrict to the shared core emotions (see config.py).
    if dataset_key == "cameo":
        raw, all_labels = filter_labels(raw, base_split, label_col, CAMEO_CORE_EMOTIONS)

    label_names, id2label, label2id, num_labels = build_label_maps(
        raw, base_split, label_col, all_labels
    )
    # CAMEO also gets language-aware stratification so per-language test
    # proportions (and thus per-language F1-macro) stay meaningful.
    extra_stratify_col = "language" if dataset_key == "cameo" else None

    # Decode the whole dataset once (expensive: reads/resamples raw audio) and
    # featurise it for this model (cheap: per-model normalization only). The
    # HPO split and every CV fold below select() their rows out of full_ds by
    # index rather than each re-running audio feature extraction.
    #
    # decode_split() doesn't close over anything model-specific, so its HF
    # datasets cache fingerprint is the same for every model in MODELS — the
    # first model to run pays for the raw-audio read, every later model for
    # the same dataset hits the on-disk cache instead of re-decoding.
    feature_extractor = load_feature_extractor(hf_model)
    decoded_ds = decode_split(raw[base_split], label_col, all_labels, label2id, desc="Decoding")
    full_ds = featurise_split(decoded_ds, feature_extractor, desc="Featurising")

    # ── Hyperparameter selection ───────────────────────────────────────────────
    # Priority:
    #   1. best_hyperparameters.json already on disk (from a prior HPO-only run)
    #   2. Run Optuna HPO now, on a single held-out split — tuned once, not per
    #      CV fold
    #   3. Fall back to the config defaults and persist them for reproducibility
    hpo_params_path = os.path.join(output_dir, "best_hyperparameters.json")
    warmup = WARMUP_RATIO
    decay  = WEIGHT_DECAY

    if os.path.exists(hpo_params_path):
        with open(hpo_params_path) as f:
            saved = json.load(f)
        lr         = saved.get("learning_rate",               lr)
        batch_size = int(saved.get("per_device_train_batch_size", batch_size))
        warmup     = saved.get("warmup_ratio",                warmup)
        decay      = saved.get("weight_decay",                decay)
        print(
            f"  Loaded hyperparameters: lr={lr}  bs={batch_size}  "
            f"warmup={warmup}  wd={decay}"
        )

    elif hpo_trials > 0:
        hpo_train_idx, hpo_val_idx, _ = split_dataset(
            raw, base_split, label_col, all_labels, seed, extra_stratify_col=extra_stratify_col
        )
        # Subsample large train splits (e.g. CAMEO) so Stage-1 search doesn't
        # dominate the wall-clock budget.
        hpo_train_ds = subsample_for_hpo(
            full_ds.select(hpo_train_idx), HPO_SUBSAMPLE_THRESHOLD, HPO_SUBSAMPLE_FRACTION, seed
        )
        hpo_val_ds = full_ds.select(hpo_val_idx)
        best = run_hpo(
            hpo_train_ds, hpo_val_ds, hf_model, num_labels, id2label, label2id,
            output_dir, n_trials=hpo_trials,
        )
        lr         = best["learning_rate"]
        batch_size = int(best["per_device_train_batch_size"])
        warmup     = best.get("warmup_ratio", warmup)
        decay      = best.get("weight_decay",  decay)

    else:
        _save_default_hparams(hpo_params_path, lr, batch_size, warmup, decay,
                              note="defaults — HPO disabled")

    # ── HPO-only mode: stop here after saving best_hyperparameters.json ───────
    if hpo_only:
        print("  HPO complete. Skipping training (--hpo_only mode).")
        return {}

    # ── K-fold CV: train + evaluate a fresh model per fold ─────────────────────
    fold_metrics = []
    fold_lang_metrics = []
    for fold_i, train_idx, val_idx, test_idx in kfold_split_dataset(
        raw, base_split, label_col, all_labels, seed,
        n_splits=k_folds, extra_stratify_col=extra_stratify_col,
    ):
        fold_dir = os.path.join(output_dir, f"fold_{fold_i}")
        fold_metrics_path = os.path.join(fold_dir, "metrics.json")

        if skip_done and os.path.exists(fold_metrics_path):
            print(f"  Fold {fold_i + 1}/{k_folds}: SKIP (metrics.json exists)")
            with open(fold_metrics_path) as f:
                fold_metrics.append(json.load(f))
        else:
            os.makedirs(fold_dir, exist_ok=True)
            print(f"\n  --- Fold {fold_i + 1}/{k_folds} ---")

            train_ds = full_ds.select(train_idx)
            val_ds   = full_ds.select(val_idx)
            test_ds  = full_ds.select(test_idx)
            test_raw_fold = raw[base_split].select(test_idx)

            model = load_model(hf_model, num_labels, id2label, label2id)
            trainer = train_model(
                train_ds, val_ds, model, fold_dir,
                lr=lr, batch_size=batch_size,
                warmup_ratio=warmup, weight_decay=decay,
                epochs=epochs,
            )

            pred_output = run_inference(trainer, test_ds)

            m = save_results(pred_output, test_raw_fold, id2label, num_labels, fold_dir)
            save_training_curves(trainer, fold_dir)
            fold_metrics.append(m)

            # metrics/predictions/curves above are everything downstream code
            # needs — the Trainer checkpoint (model + optimizer + scheduler
            # state, kept via save_strategy="best") is dead weight once eval
            # is done. Left alone it accumulates forever, one per fold, and
            # for large models is the main driver of disk-quota exhaustion.
            for ckpt_dir in glob.glob(os.path.join(fold_dir, "checkpoint-*")):
                shutil.rmtree(ckpt_dir, ignore_errors=True)

        lang_path = os.path.join(fold_dir, "metrics_by_language.json")
        if os.path.exists(lang_path):
            with open(lang_path) as f:
                fold_lang_metrics.append(json.load(f))

    metrics = {}
    if fold_metrics:
        metrics = aggregate_cv_metrics(fold_metrics, output_dir)
        aggregate_cv_language_metrics(fold_lang_metrics, output_dir)

    return metrics


def _save_default_hparams(path, lr, batch_size, warmup, decay, note=""):
    data = {
        "learning_rate":               lr,
        "per_device_train_batch_size": batch_size,
        "warmup_ratio":                warmup,
        "weight_decay":                decay,
    }
    if note:
        data["note"] = note
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run all model × dataset experiments.")
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LEARNING_RATE)
    parser.add_argument("--hpo_trials",  type=int,   default=HPO_TRIALS)
    parser.add_argument("--k_folds",     type=int,   default=K_FOLDS,
                         help="Number of stratified CV folds in Stage 2")
    parser.add_argument("--seed",        type=int,   default=SEED)
    parser.add_argument(
        "--models", nargs="*", choices=list(MODELS.keys()), default=list(MODELS.keys()),
        help="Subset of models to run (default: all)",
    )
    parser.add_argument(
        "--datasets", nargs="*", choices=list(DATASETS.keys()), default=list(DATASETS.keys()),
        help="Subset of datasets to run (default: all)",
    )
    parser.add_argument(
        "--skip_done", action="store_true",
        help="Skip experiments with cv_metrics.json, and folds with fold_*/metrics.json, already done",
    )
    parser.add_argument(
        "--hpo_only", action="store_true",
        help="Stage 1: run Optuna HPO only, save best_hyperparameters.json, skip training",
    )
    args = parser.parse_args()

    stage_label = "Stage 1 — HPO only" if args.hpo_only else "Stage 2 — Training"

    print("=" * 70)
    print(f"Emotion Recognition Pipeline  [{stage_label}]")
    print("=" * 70)
    print(f"Device    : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Models    : {args.models}")
    print(f"Datasets  : {args.datasets}")
    if args.hpo_only:
        print(f"HPO trials: {args.hpo_trials}  |  Seed: {args.seed}")
    else:
        print(f"Epochs    : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}  |  K-folds: {args.k_folds}")
    print(f"Output    : {args.output_dir}")
    print("=" * 70)
    os.makedirs(args.output_dir, exist_ok=True)

    combos = [
        (mk, MODELS[mk], dk, DATASETS[dk])
        for mk in args.models
        for dk in args.datasets
    ]
    all_results: list[dict] = []

    for i, (model_key, hf_model, dataset_key, hf_dataset) in enumerate(combos, 1):
        exp_name   = f"{model_key}_{dataset_key}"
        output_dir = os.path.join(args.output_dir, exp_name)

        # Skip already-completed (all folds done) training experiments
        cv_metrics_path = os.path.join(output_dir, "cv_metrics.json")
        if args.skip_done and not args.hpo_only and os.path.exists(cv_metrics_path):
            print(f"\n[{i}/{len(combos)}] SKIP {exp_name}  (cv_metrics.json exists)")
            all_results.append({"model": model_key, "dataset": dataset_key, **load_cv_metrics(output_dir)})
            continue

        # Skip HPO if best_hyperparameters.json already exists
        hpo_params_path = os.path.join(output_dir, "best_hyperparameters.json")
        if args.hpo_only and os.path.exists(hpo_params_path):
            print(f"\n[{i}/{len(combos)}] SKIP HPO for {exp_name}  (best_hyperparameters.json exists)")
            continue

        print(f"\n{'='*70}")
        print(f"[{i}/{len(combos)}]  {exp_name}")
        print(f"  Model  : {hf_model}")
        print(f"  Dataset: {hf_dataset}")
        print(f"  Output : {output_dir}")

        try:
            metrics = run_experiment(
                model_key=model_key,
                hf_model=hf_model,
                dataset_key=dataset_key,
                hf_dataset=hf_dataset,
                output_dir=output_dir,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                hpo_trials=args.hpo_trials,
                k_folds=args.k_folds,
                seed=args.seed,
                hpo_only=args.hpo_only,
                skip_done=args.skip_done,
            )
            if not args.hpo_only:
                all_results.append({"model": model_key, "dataset": dataset_key, **metrics})

        except Exception:
            tb  = traceback.format_exc()
            err = tb.strip().splitlines()[-1]
            print(f"\nERROR in {exp_name}:\n{tb}")
            if not args.hpo_only:
                all_results.append({"model": model_key, "dataset": dataset_key, "error": err})

    if all_results:
        generate_report(all_results, args.output_dir)

    if args.hpo_only:
        print("\nStage 1 complete. Run Stage 2 to train with the selected hyperparameters.")
    else:
        print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
