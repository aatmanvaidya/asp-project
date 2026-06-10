import os
import warnings
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import datasets.config
datasets.config.TORCHCODEC_AVAILABLE = False

from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings('ignore')


# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME    = 'facebook/wav2vec2-base'
DATASET_NAME  = 'xbgoose/ravdess'

SAMPLING_RATE = 16_000
MAX_DURATION  = 5.0
MAX_LENGTH    = int(SAMPLING_RATE * MAX_DURATION)

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15

BATCH_SIZE    = 8
NUM_EPOCHS    = 10
LEARNING_RATE = 3e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
OUTPUT_DIR    = './wav2vec2-ravdess-output'
SEED          = 42

RAVDESS_NAMES = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Wav2Vec2 for RAVDESS emotion classification')
    parser.add_argument('--model_name',    type=str,   default=MODEL_NAME)
    parser.add_argument('--dataset_name',  type=str,   default=DATASET_NAME)
    parser.add_argument('--batch_size',    type=int,   default=BATCH_SIZE)
    parser.add_argument('--num_epochs',    type=int,   default=NUM_EPOCHS)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight_decay',  type=float, default=WEIGHT_DECAY)
    parser.add_argument('--warmup_ratio',  type=float, default=WARMUP_RATIO)
    parser.add_argument('--output_dir',    type=str,   default=OUTPUT_DIR)
    parser.add_argument('--seed',          type=int,   default=SEED)
    parser.add_argument('--max_duration',  type=float, default=MAX_DURATION)
    return parser.parse_args()


def load_and_explore(dataset_name):
    print('\n[1/7] Loading dataset ...')
    raw_dataset = load_dataset(dataset_name)
    print(raw_dataset)

    base_split = list(raw_dataset.keys())[0]
    example    = raw_dataset[base_split][0]

    print('Available splits :', list(raw_dataset.keys()))
    print('Columns          :', list(example.keys()))

    label_col = next(
        (c for c in ['label', 'emotion', 'labels', 'target'] if c in example),
        None,
    )
    assert label_col is not None, 'Could not detect label column.'
    print(f'Label column     : "{label_col}"')

    all_labels = raw_dataset[base_split][label_col]
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f'Unique labels : {unique}')
    print(f'Total samples : {len(all_labels):,}')
    for u, c in zip(unique, counts):
        print(f'  {u:>12} -> {c} samples')

    return raw_dataset, base_split, label_col, all_labels


def build_label_maps(raw_dataset, base_split, label_col, all_labels):
    print('\n[2/7] Building label maps ...')
    features = raw_dataset[base_split].features
    if hasattr(features.get(label_col), 'names'):
        label_names = features[label_col].names
        print('Using dataset ClassLabel names:', label_names)
    else:
        min_label = int(min(all_labels))
        n_labels  = int(max(all_labels)) - min_label + 1
        label_names = RAVDESS_NAMES[:n_labels]
        if min_label != 0:
            print(f'Labels start at {min_label}; will shift to 0-indexed during preprocessing.')
        print('Using default RAVDESS names:', label_names)

    id2label   = {i: name for i, name in enumerate(label_names)}
    label2id   = {name: i for i, name in enumerate(label_names)}
    num_labels = len(label_names)
    print(f'id2label   : {id2label}')
    print(f'Num classes: {num_labels}')
    return label_names, id2label, label2id, num_labels


def split_dataset(raw_dataset, base_split, label_col, all_labels, seed):
    print('\n[3/7] Splitting dataset (70/15/15) ...')
    base_data = raw_dataset[base_split]
    indices   = list(range(len(base_data)))

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
    val_raw   = base_data.select(val_idx)
    test_raw  = base_data.select(test_idx)

    n = len(base_data)
    print(f'Train : {len(train_raw):>5,} samples  ({len(train_raw)/n*100:.1f}%)')
    print(f'Val   : {len(val_raw):>5,} samples  ({len(val_raw)/n*100:.1f}%)')
    print(f'Test  : {len(test_raw):>5,} samples  ({len(test_raw)/n*100:.1f}%)')
    return train_raw, val_raw, test_raw


def preprocess_splits(train_raw, val_raw, test_raw, label_col, all_labels, model_name, max_length):
    print('\n[4/7] Extracting features ...')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    print('Feature extractor sampling rate:', feature_extractor.sampling_rate)

    train_raw = train_raw.cast_column('audio', Audio(sampling_rate=SAMPLING_RATE))
    val_raw   = val_raw.cast_column(  'audio', Audio(sampling_rate=SAMPLING_RATE))
    test_raw  = test_raw.cast_column( 'audio', Audio(sampling_rate=SAMPLING_RATE))

    _min_label = int(min(all_labels))

    def preprocess(batch):
        audio_arrays = [x['array'] for x in batch['audio']]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=SAMPLING_RATE,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
        )
        inputs['labels'] = [int(l) - _min_label for l in batch[label_col]]
        return inputs

    cols_to_remove = train_raw.column_names
    train_ds = train_raw.map(preprocess, batched=True, remove_columns=cols_to_remove, desc='Preprocessing train')
    val_ds   = val_raw.map(  preprocess, batched=True, remove_columns=cols_to_remove, desc='Preprocessing val  ')
    test_ds  = test_raw.map( preprocess, batched=True, remove_columns=cols_to_remove, desc='Preprocessing test ')

    train_ds.set_format('torch')
    val_ds.set_format(  'torch')
    test_ds.set_format( 'torch')

    print('Columns after preprocessing:', train_ds.column_names)
    print('input_values shape (first sample):', train_ds[0]['input_values'].shape)
    print('Sample label:', train_ds[0]['labels'].item())
    return train_ds, val_ds, test_ds


def load_model(model_name, num_labels, id2label, label2id):
    print('\n[5/7] Loading model ...')
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters    : {total:,}')
    print(f'Trainable parameters: {trainable:,}')
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'f1_macro'   : f1_score(labels, preds, average='macro',    zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        'accuracy'   : float((preds == labels).mean()),
    }


def train(model, train_ds, val_ds, args):
    print('\n[6/7] Training ...')
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to='none',
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
    print('\nTraining complete.')
    print(f'Runtime  : {train_result.metrics["train_runtime"]:.1f}s')
    print(f'Samples/s: {train_result.metrics["train_samples_per_second"]:.2f}')
    return trainer


def evaluate_test(trainer, test_ds, id2label, num_labels, output_dir):
    print('\n[7/7] Evaluating on test set ...')
    test_output = trainer.predict(test_ds)
    test_preds  = np.argmax(test_output.predictions, axis=-1)
    test_labels = test_output.label_ids
    label_names = [id2label[i] for i in range(num_labels)]

    print(f'Test samples: {len(test_labels):,}')
    print('=' * 64)
    print('CLASSIFICATION REPORT')
    print('=' * 64)
    print(classification_report(test_labels, test_preds, target_names=label_names, zero_division=0))

    f1_macro    = f1_score(test_labels, test_preds, average='macro',    zero_division=0)
    f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    accuracy    = float((test_preds == test_labels).mean())
    print(f'F1-Macro    : {f1_macro:.4f}')
    print(f'F1-Weighted : {f1_weighted:.4f}')
    print(f'Accuracy    : {accuracy:.4f}')

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f'Confusion Matrix — Test Set  (F1-Macro={f1_macro:.3f})', fontsize=14, pad=14)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f'Saved -> {cm_path}')

    # Training curves
    log_history  = trainer.state.log_history
    train_steps  = [e['step'] for e in log_history if 'loss' in e and 'eval_loss' not in e]
    train_losses = [e['loss'] for e in log_history if 'loss' in e and 'eval_loss' not in e]
    eval_epochs  = [e['epoch']     for e in log_history if 'eval_loss' in e]
    eval_losses  = [e['eval_loss'] for e in log_history if 'eval_loss' in e]
    eval_f1      = [e.get('eval_f1_macro') for e in log_history if 'eval_loss' in e]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_steps, train_losses, color='steelblue', alpha=0.85, linewidth=1.2)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    ax_loss = axes[1]
    ax_loss.plot(eval_epochs, eval_losses, color='coral', marker='o', label='Val Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss', color='coral')
    ax_loss.tick_params(axis='y', labelcolor='coral')
    ax_loss.set_title('Validation Loss & F1-Macro per Epoch')

    if any(v is not None for v in eval_f1):
        ax_f1 = ax_loss.twinx()
        ax_f1.plot(eval_epochs, eval_f1, color='seagreen', marker='s', linestyle='--', label='Val F1-Macro')
        ax_f1.set_ylabel('F1-Macro', color='seagreen')
        ax_f1.tick_params(axis='y', labelcolor='seagreen')
        lines  = ax_loss.get_lines() + ax_f1.get_lines()
        labels = [l.get_label() for l in lines]
        ax_loss.legend(lines, labels, loc='upper right')
    else:
        ax_loss.legend(loc='upper right')

    ax_loss.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f'Saved -> {curves_path}')


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device    : {device}')
    print(f'Model     : {args.model_name}')
    print(f'Dataset   : {args.dataset_name}')
    print(f'Epochs    : {args.num_epochs}')
    print(f'Batch size: {args.batch_size}')
    print(f'LR        : {args.learning_rate}')
    max_length = int(SAMPLING_RATE * args.max_duration)
    print(f'Max length: {max_length:,} samples ({args.max_duration}s @ {SAMPLING_RATE} Hz)')

    os.makedirs(args.output_dir, exist_ok=True)

    raw_dataset, base_split, label_col, all_labels = load_and_explore(args.dataset_name)
    label_names, id2label, label2id, num_labels    = build_label_maps(raw_dataset, base_split, label_col, all_labels)
    train_raw, val_raw, test_raw                   = split_dataset(raw_dataset, base_split, label_col, all_labels, args.seed)
    train_ds, val_ds, test_ds                      = preprocess_splits(train_raw, val_raw, test_raw, label_col, all_labels, args.model_name, max_length)
    model                                          = load_model(args.model_name, num_labels, id2label, label2id)
    trainer                                        = train(model, train_ds, val_ds, args)
    evaluate_test(trainer, test_ds, id2label, num_labels, args.output_dir)


if __name__ == '__main__':
    main()
