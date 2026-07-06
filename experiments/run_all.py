"""
Emotion recognition pipeline — runs all model × dataset combinations.

Two-stage workflow (see run_pipeline.sh):

  Stage 1 — HPO (single GPU, Optuna finds best hyperparameters per experiment):
    python experiments/run_all.py --hpo_trials 20 --hpo_only

  Stage 2 — Full training (4-GPU DDP, loads Stage-1 hyperparameters):
    torchrun --standalone --nproc_per_node=4 experiments/run_all.py --skip_done

Single-GPU end-to-end (HPO + training in one go):
    python experiments/run_all.py --hpo_trials 20

Include MFCC baseline (runs on CPU, no HPO needed):
    python experiments/run_all.py --include_mfcc --datasets ravdess cameo

Additional flags:
    --output_dir   outputs/         root directory for all experiment outputs
    --epochs       30               max training epochs (early stopping may stop sooner)
    --mfcc_epochs  50               epochs for the MFCC CNN baseline
    --batch_size   8
    --lr           3e-5
    --seed         42
    --models       wav2vec2-base …  subset of models to run (default: all)
    --datasets     ravdess cameo …  subset of datasets to run (default: all)
    --skip_done                     skip experiments that already have metrics.json
    --hpo_only                      run HPO only, save best_hyperparameters.json, skip training
    --include_mfcc                  also run the MFCC + 1D CNN baseline for each dataset
"""
import argparse
import json
import os
import shutil
import sys
import time
import traceback

# ── Audio backend must be patched before ANY datasets/audio import ─────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline.audio_backend  # noqa: F401, E402  (side-effect import — must be first)

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402

from pipeline.config import (  # noqa: E402
    BATCH_SIZE,
    DATASETS,
    HPO_TRIALS,
    LEARNING_RATE,
    MODELS,
    NUM_EPOCHS,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from pipeline.data import (  # noqa: E402
    build_label_maps,
    load_dataset_hf,
    preprocess_splits,
    split_dataset,
)
from pipeline.evaluate import run_inference, save_results, save_training_curves  # noqa: E402
from pipeline.model import load_feature_extractor, load_model  # noqa: E402
from pipeline.report import generate_report  # noqa: E402
from pipeline.trainer_utils import run_hpo, train_model  # noqa: E402

# Root of the repository (one level above experiments/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_main() -> bool:
    """True on rank 0 and when not running under torchrun."""
    return int(os.environ.get("LOCAL_RANK", -1)) <= 0


def _wait_for_file(path: str, poll_seconds: float = 5.0, timeout_seconds: float = 3600.0) -> None:
    """Block until `path` exists, raising if the main rank never produces it."""
    waited = 0.0
    while not os.path.exists(path):
        time.sleep(poll_seconds)
        waited += poll_seconds
        if waited >= timeout_seconds:
            raise TimeoutError(f"Timed out after {timeout_seconds:.0f}s waiting for {path}")


def _load_or_build_splits(hf_model: str, hf_dataset: str, seed: int, output_dir: str):
    """Build train/val/test feature datasets once, cached under output_dir.

    Under DDP (torchrun) every rank imports and calls this function, but only
    rank 0 downloads/decodes/feature-extracts the raw audio — that pipeline is
    expensive enough that doing it redundantly on all ranks at once has caused
    the whole node to OOM. Other ranks wait for the cache and load it from
    disk instead. This also means Stage 1 (single-GPU) primes the cache that
    Stage 2 (4-GPU DDP) then reuses almost for free.
    """
    cache_dir     = os.path.join(output_dir, "_features")
    ready_marker  = os.path.join(cache_dir, "READY")
    label_map_path = os.path.join(cache_dir, "label_maps.json")

    if _is_main() and not os.path.exists(ready_marker):
        # Clear out any partial cache left by a job that died mid-write.
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        raw, base_split, label_col, all_labels = load_dataset_hf(hf_dataset)
        label_names, id2label, label2id, num_labels = build_label_maps(
            raw, base_split, label_col, all_labels
        )
        train_raw, val_raw, test_raw = split_dataset(raw, base_split, label_col, all_labels, seed)
        feature_extractor = load_feature_extractor(hf_model)
        train_ds, val_ds, test_ds = preprocess_splits(
            train_raw, val_raw, test_raw, label_col, all_labels, label2id, feature_extractor
        )

        os.makedirs(cache_dir, exist_ok=True)
        train_ds.save_to_disk(os.path.join(cache_dir, "train"))
        val_ds.save_to_disk(os.path.join(cache_dir, "val"))
        test_ds.save_to_disk(os.path.join(cache_dir, "test"))
        test_raw.save_to_disk(os.path.join(cache_dir, "test_raw"))
        with open(label_map_path, "w") as f:
            json.dump({"id2label": id2label, "label2id": label2id, "num_labels": num_labels}, f)
        # Written last: its existence is the signal to other ranks that the cache is complete.
        with open(ready_marker, "w") as f:
            f.write("ok")

    elif not _is_main():
        _wait_for_file(ready_marker)

    train_ds = load_from_disk(os.path.join(cache_dir, "train")).with_format("torch")
    val_ds   = load_from_disk(os.path.join(cache_dir, "val")).with_format("torch")
    test_ds  = load_from_disk(os.path.join(cache_dir, "test")).with_format("torch")
    test_raw = load_from_disk(os.path.join(cache_dir, "test_raw"))
    with open(label_map_path) as f:
        maps = json.load(f)
    id2label   = {int(k): v for k, v in maps["id2label"].items()}
    label2id   = maps["label2id"]
    num_labels = maps["num_labels"]
    return train_ds, val_ds, test_ds, test_raw, id2label, label2id, num_labels


def run_mfcc_experiment(
    dataset_key: str,
    output_dir: str,
    epochs: int = 50,
    seed: int = SEED,
) -> dict:
    """Run the MFCC + 1D CNN baseline for a given dataset.

    Uses data/<dataset_key>.py to load raw audio records, then calls
    baselines/train.py:run() to train and evaluate.  Results (metrics.json,
    confusion_matrix.png, etc.) are written to output_dir.
    """
    import importlib

    sys.path.insert(0, _REPO_ROOT)

    data_module = importlib.import_module(f"data.{dataset_key}")
    print(f"\n  Loading {dataset_key} records for MFCC baseline ...")
    records = data_module.load()

    from baselines.features import build_feature_matrix  # noqa: E402
    import baselines.train as mfcc_train  # noqa: E402

    print(f"  Extracting MFCC features for {len(records):,} samples ...")
    X, y, label_names = build_feature_matrix(records)

    print(f"  Training MFCC CNN for {epochs} epochs ...")
    metrics = mfcc_train.run(
        X, y, label_names,
        epochs=epochs,
        output_dir=output_dir,
    )
    return metrics or {}


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
    seed: int       = SEED,
    hpo_only: bool  = False,
) -> dict:
    main = _is_main()
    if main:
        os.makedirs(output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, test_raw, id2label, label2id, num_labels = _load_or_build_splits(
        hf_model, hf_dataset, seed, output_dir
    )

    # ── Hyperparameter selection ───────────────────────────────────────────────
    # Priority:
    #   1. best_hyperparameters.json already on disk (from a prior HPO-only run)
    #   2. Run Optuna HPO now (only on a single-GPU / non-DDP process)
    #   3. Fall back to the config defaults and persist them for reproducibility
    hpo_params_path = os.path.join(output_dir, "best_hyperparameters.json")
    using_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    warmup = WARMUP_RATIO
    decay  = WEIGHT_DECAY

    if os.path.exists(hpo_params_path):
        with open(hpo_params_path) as f:
            saved = json.load(f)
        lr         = saved.get("learning_rate",               lr)
        batch_size = int(saved.get("per_device_train_batch_size", batch_size))
        warmup     = saved.get("warmup_ratio",                warmup)
        decay      = saved.get("weight_decay",                decay)
        if main:
            print(
                f"  Loaded hyperparameters: lr={lr}  bs={batch_size}  "
                f"warmup={warmup}  wd={decay}"
            )

    elif hpo_trials > 0:
        if using_ddp:
            # HPO is not compatible with DDP — save defaults and warn
            if main:
                print(
                    "  WARNING: HPO cannot run under DDP (WORLD_SIZE > 1). "
                    "Run Stage 1 (--hpo_only) on a single GPU first."
                )
                _save_default_hparams(hpo_params_path, lr, batch_size, warmup, decay,
                                      note="defaults — HPO skipped (DDP mode)")
        elif main:
            # Single-GPU HPO
            best = run_hpo(
                train_ds, val_ds, hf_model, num_labels, id2label, label2id,
                output_dir, n_trials=hpo_trials,
            )
            lr         = best["learning_rate"]
            batch_size = int(best["per_device_train_batch_size"])
            warmup     = best.get("warmup_ratio", warmup)
            decay      = best.get("weight_decay",  decay)

    else:
        if main:
            _save_default_hparams(hpo_params_path, lr, batch_size, warmup, decay,
                                  note="defaults — HPO disabled")

    # ── HPO-only mode: stop here after saving best_hyperparameters.json ───────
    if hpo_only:
        if main:
            print("  HPO complete. Skipping training (--hpo_only mode).")
        return {}

    # ── Model + training ──────────────────────────────────────────────────────
    model = load_model(hf_model, num_labels, id2label, label2id)
    trainer = train_model(
        train_ds, val_ds, model, output_dir,
        lr=lr, batch_size=batch_size,
        warmup_ratio=warmup, weight_decay=decay,
        epochs=epochs,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    # run_inference is called from ALL ranks (DDP gather happens inside Trainer)
    pred_output = run_inference(trainer, test_ds)

    metrics = {}
    if main:
        metrics = save_results(pred_output, test_raw, id2label, num_labels, output_dir)
        save_training_curves(trainer, output_dir)

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
    parser.add_argument("--output_dir",   default="outputs")
    parser.add_argument("--epochs",       type=int,   default=NUM_EPOCHS)
    parser.add_argument("--mfcc_epochs",  type=int,   default=50,
                        help="Training epochs for the MFCC CNN baseline")
    parser.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",           type=float, default=LEARNING_RATE)
    parser.add_argument("--hpo_trials",   type=int,   default=HPO_TRIALS)
    parser.add_argument("--seed",         type=int,   default=SEED)
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
        help="Skip experiments that already have metrics.json",
    )
    parser.add_argument(
        "--hpo_only", action="store_true",
        help="Stage 1: run Optuna HPO only, save best_hyperparameters.json, skip training",
    )
    parser.add_argument(
        "--include_mfcc", action="store_true",
        help="Also run the MFCC + 1D CNN baseline for each dataset (CPU, no HPO)",
    )
    args = parser.parse_args()

    main_proc = _is_main()
    n_gpus = torch.cuda.device_count()
    stage_label = "Stage 1 — HPO only" if args.hpo_only else "Stage 2 — Training"

    if main_proc:
        print("=" * 70)
        print(f"Emotion Recognition Pipeline  [{stage_label}]")
        print("=" * 70)
        print(f"Device    : {'cuda' if torch.cuda.is_available() else 'cpu'}  ({n_gpus} GPU(s))")
        print(f"Models    : {args.models}")
        print(f"Datasets  : {args.datasets}")
        if args.hpo_only:
            print(f"HPO trials: {args.hpo_trials}  |  Seed: {args.seed}")
        else:
            print(f"Epochs    : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
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

        # Skip already-completed training experiments
        metrics_path = os.path.join(output_dir, "metrics.json")
        if args.skip_done and not args.hpo_only and os.path.exists(metrics_path):
            if main_proc:
                print(f"\n[{i}/{len(combos)}] SKIP {exp_name}  (metrics.json exists)")
            with open(metrics_path) as f:
                cached = json.load(f)
            all_results.append({"model": model_key, "dataset": dataset_key, **cached})
            continue

        # Skip HPO if best_hyperparameters.json already exists
        hpo_params_path = os.path.join(output_dir, "best_hyperparameters.json")
        if args.hpo_only and os.path.exists(hpo_params_path):
            if main_proc:
                print(f"\n[{i}/{len(combos)}] SKIP HPO for {exp_name}  (best_hyperparameters.json exists)")
            continue

        if main_proc:
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
                seed=args.seed,
                hpo_only=args.hpo_only,
            )
            if not args.hpo_only:
                all_results.append({"model": model_key, "dataset": dataset_key, **metrics})

        except Exception:
            tb  = traceback.format_exc()
            err = tb.strip().splitlines()[-1]
            if main_proc:
                print(f"\nERROR in {exp_name}:\n{tb}")
            if not args.hpo_only:
                all_results.append({"model": model_key, "dataset": dataset_key, "error": err})

    # ── MFCC baseline (CPU-only, no DDP, no HPO) ─────────────────────────────
    if main_proc and args.include_mfcc and not args.hpo_only:
        mfcc_datasets = args.datasets
        print(f"\n{'='*70}")
        print(f"MFCC + 1D CNN Baseline  (datasets: {mfcc_datasets})")
        print(f"{'='*70}")

        for dataset_key in mfcc_datasets:
            exp_name   = f"mfcc-cnn_{dataset_key}"
            output_dir = os.path.join(args.output_dir, exp_name)
            metrics_path = os.path.join(output_dir, "metrics.json")

            if args.skip_done and os.path.exists(metrics_path):
                print(f"\nSKIP {exp_name}  (metrics.json exists)")
                with open(metrics_path) as f:
                    cached = json.load(f)
                all_results.append({"model": "mfcc-cnn", "dataset": dataset_key, **cached})
                continue

            print(f"\n{'='*70}")
            print(f"  {exp_name}")
            print(f"  Output : {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            try:
                metrics = run_mfcc_experiment(
                    dataset_key=dataset_key,
                    output_dir=output_dir,
                    epochs=args.mfcc_epochs,
                    seed=args.seed,
                )
                all_results.append({"model": "mfcc-cnn", "dataset": dataset_key, **metrics})
            except Exception:
                tb  = traceback.format_exc()
                err = tb.strip().splitlines()[-1]
                print(f"\nERROR in {exp_name}:\n{tb}")
                all_results.append({"model": "mfcc-cnn", "dataset": dataset_key, "error": err})

    if main_proc and all_results:
        generate_report(all_results, args.output_dir)

    if main_proc:
        if args.hpo_only:
            print("\nStage 1 complete. Run Stage 2 to train with the selected hyperparameters.")
        else:
            print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
