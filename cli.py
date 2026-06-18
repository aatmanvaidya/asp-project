#!/usr/bin/env python3
"""Driver CLI for ASP emotion classification.

Usage:
    python cli.py train mfcc        --dataset ravdess [--epochs 50] [--lr 1e-3] [--batch-size 32]
    python cli.py train transformer --dataset ravdess [--model-id facebook/wav2vec2-base] ...
    python cli.py evaluate          --dataset ravdess  --model models/mfcc_cnn.pt

Data loaders live in data/<name>.py and must expose:
    def load() -> list[dict]:
        # each dict: {"array": np.ndarray, "sr": int, "label": str}
"""

import argparse
import importlib.util
import os
import sys


def _load_data_loader(name: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"{name}.py")
    if not os.path.exists(path):
        print(f"Error: no data loader found for dataset '{name}'.")
        print(f"  Create data/{name}.py with a load() -> list[dict] function.")
        print(f"  Each dict must have keys: 'array' (np.ndarray), 'sr' (int), 'label' (str).")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location(f"data.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- command handlers ---

def cmd_train_mfcc(args):
    loader = _load_data_loader(args.dataset)
    records = loader.load()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines"))
    from features import build_feature_matrix
    from train import run

    X, y, label_names = build_feature_matrix(records)
    run(X, y, label_names, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)


def cmd_train_transformer(args):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"))
    import wav2vec2_ravdess_emotion as pipeline

    pipeline.run(
        dataset_name=args.dataset,
        model_name=args.model_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hpo_trials=args.hpo_trials,
    )


def cmd_evaluate(args):
    raise NotImplementedError(
        "Standalone evaluation is not yet implemented. "
        "Add it under baselines/ and wire it up here."
    )


# --- parsers ---

def _add_training_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, metavar="NAME",
                   help="Dataset name — loads data/<name>.py")
    p.add_argument("--epochs", type=int, default=50, metavar="N")
    p.add_argument("--lr", type=float, default=1e-3, metavar="F")
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size", metavar="N")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="ASP emotion classification — training and evaluation driver",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_sub = train_parser.add_subparsers(dest="model_type", required=True)

    # train mfcc
    mfcc_p = train_sub.add_parser("mfcc", help="MFCC + 1D CNN baseline")
    _add_training_args(mfcc_p)
    mfcc_p.set_defaults(func=cmd_train_mfcc)

    # train transformer
    tf_p = train_sub.add_parser("transformer", help="Fine-tune a wav2vec2 transformer")
    _add_training_args(tf_p)
    tf_p.add_argument(
        "--model-id", dest="model_id", default="facebook/wav2vec2-base", metavar="HF_ID",
        help="HuggingFace model ID (default: facebook/wav2vec2-base)",
    )
    tf_p.add_argument(
        "--output-dir", dest="output_dir", default="experiments/wav2vec2-ravdess-output",
        metavar="DIR", help="Directory for checkpoints and outputs",
    )
    tf_p.add_argument(
        "--hpo-trials", dest="hpo_trials", type=int, default=20, metavar="N",
        help="Optuna HPO trials (default: 20, 0 to disable)",
    )
    tf_p.set_defaults(func=cmd_train_transformer)

    # --- evaluate ---
    eval_p = subparsers.add_parser("evaluate", help="Evaluate a trained model [not implemented]")
    eval_p.add_argument("--dataset", required=True, metavar="NAME",
                        help="Dataset name — loads data/<name>.py")
    eval_p.add_argument("--model", required=True, metavar="PATH",
                        help="Path to saved model weights")
    eval_p.set_defaults(func=cmd_evaluate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
