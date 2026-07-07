"""Dataset loading, stratified splitting, and feature extraction."""
import functools

import numpy as np
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from .config import (
    MAX_LENGTH,
    RAVDESS_EMOTION_NAMES,
    SAMPLING_RATE,
    TEST_RATIO,
    VAL_RATIO,
)


def load_dataset_hf(hf_name: str):
    """Load a HuggingFace dataset and return split info and raw labels."""
    print(f"\n  Loading: {hf_name}")
    raw = load_dataset(hf_name)

    if len(raw) > 1:
        # Some datasets (e.g. CAMEO) ship one HF split per sub-corpus rather
        # than a train/test split — merge them into a single working split.
        print(f"  Multiple splits found ({list(raw.keys())}) — concatenating into one.")
        merged = concatenate_datasets([raw[s] for s in raw.keys()])
        raw = DatasetDict({"all": merged})

    base_split = list(raw.keys())[0]
    features = raw[base_split].features

    label_col = next(
        (c for c in ["label", "emotion", "labels", "target"] if c in features), None
    )
    assert label_col, f"No label column found in {list(features.keys())}"

    all_labels = raw[base_split].with_format("numpy")[label_col]
    unique, counts = np.unique(all_labels, return_counts=True)
    print(f"  Label col : {label_col}")
    print(f"  Samples   : {len(all_labels):,}  |  Classes: {len(unique)}")
    for u, c in zip(unique, counts):
        print(f"    {u}: {c}")

    return raw, base_split, label_col, all_labels


def filter_labels(raw, base_split, label_col, allowed_labels: set):
    """Drop rows whose label is not in allowed_labels; returns updated (raw, all_labels)."""
    base = raw[base_split]
    before = len(base)
    base = base.filter(lambda ex: ex[label_col] in allowed_labels)
    raw[base_split] = base
    after = len(base)
    print(f"  Filtered to core emotions: {before:,} -> {after:,} samples ({before - after:,} dropped)")
    all_labels = base.with_format("numpy")[label_col]
    return raw, all_labels


def subsample_for_hpo(train_ds, threshold: int, fraction: float, seed: int):
    """
    Stratified subsample of an already-featurised train split, used only for
    Stage-1 HPO trials on large datasets. No-op if train_ds is at or below
    threshold. Does not affect the train split used for Stage-2 training.
    """
    n = len(train_ds)
    if n <= threshold:
        return train_ds

    target = max(1, int(n * fraction))
    labels = np.asarray(train_ds["labels"])
    idx = np.arange(n)
    sub_idx, _ = train_test_split(idx, train_size=target, stratify=labels, random_state=seed)
    print(f"  HPO subsample: {n:,} -> {len(sub_idx):,} train samples (stratified, {fraction:.0%})")
    return train_ds.select(sub_idx)


def build_label_maps(raw, base_split, label_col, all_labels):
    """Build id2label / label2id maps from dataset metadata or label values."""
    features = raw[base_split].features
    labels_are_strings = isinstance(all_labels[0], (str, np.str_))

    if hasattr(features.get(label_col), "names"):
        label_names = features[label_col].names
    elif labels_are_strings:
        label_names = sorted(set(str(l) for l in all_labels))  # noqa: E741
    else:
        min_l = int(min(all_labels))
        n = int(max(all_labels)) - min_l + 1
        label_names = RAVDESS_EMOTION_NAMES[:n]

    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    print(f"  Classes   : {label_names}")
    return label_names, id2label, label2id, len(label_names)


def split_dataset(raw, base_split, label_col, all_labels, seed: int, extra_stratify_col: str | None = None):
    """
    Stratified 70 / 15 / 15 split.

    If extra_stratify_col is given (e.g. "language" for CAMEO), stratifies on
    the composite (extra_col, label) key so each split preserves per-language
    label proportions. Falls back to label-only stratification if a
    language/emotion cell is too small for sklearn to split.
    """
    base = raw[base_split]
    idx = list(range(len(base)))

    strat_key = list(all_labels)
    if extra_stratify_col and extra_stratify_col in base.column_names:
        extra_vals = base.with_format("numpy")[extra_stratify_col]
        strat_key = [f"{e}|{l}" for e, l in zip(extra_vals, all_labels)]  # noqa: E741

    try:
        train_idx, tmp_idx = train_test_split(
            idx, test_size=VAL_RATIO + TEST_RATIO, stratify=strat_key, random_state=seed
        )
    except ValueError:
        print(f"  WARNING: composite stratify on '{extra_stratify_col}' + label failed "
              "(too few samples in a cell) — falling back to label-only stratify")
        strat_key = list(all_labels)
        train_idx, tmp_idx = train_test_split(
            idx, test_size=VAL_RATIO + TEST_RATIO, stratify=strat_key, random_state=seed
        )

    tmp_strat = [strat_key[i] for i in tmp_idx]
    try:
        val_idx, test_idx = train_test_split(
            tmp_idx,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            stratify=tmp_strat,
            random_state=seed,
        )
    except ValueError:
        tmp_labels = [all_labels[i] for i in tmp_idx]
        val_idx, test_idx = train_test_split(
            tmp_idx,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            stratify=tmp_labels,
            random_state=seed,
        )

    train_raw = base.select(train_idx)
    val_raw   = base.select(val_idx)
    test_raw  = base.select(test_idx)

    n = len(base)
    print(f"  Split     : {len(train_raw):,} train / {len(val_raw):,} val / {len(test_raw):,} test  (of {n:,})")
    return train_raw, val_raw, test_raw


def _preprocess_batch(
    batch,
    feature_extractor,
    label_col: str,
    label2id: dict,
    labels_are_strings: bool,
    min_label: int,
    max_length: int,
):
    all_iv, all_am = [], []
    for x in batch["audio"]:
        arr = np.array(x["array"], dtype=np.float32)
        arr = (
            arr[:max_length]
            if len(arr) >= max_length
            else np.pad(arr, (0, max_length - len(arr)))
        )
        out = feature_extractor(
            arr,
            sampling_rate=SAMPLING_RATE,
            return_attention_mask=True,
            return_tensors=None,
        )
        iv = np.array(out["input_values"], dtype=np.float32).squeeze()
        # Some extractors omit attention_mask; default to all-ones
        raw_am = out.get("attention_mask", np.ones(max_length, dtype=np.int64))
        am = np.array(raw_am, dtype=np.int64).squeeze()
        all_iv.append(iv.tolist())
        all_am.append(am.tolist())

    if labels_are_strings:
        labels = [label2id[str(l)] for l in batch[label_col]]  # noqa: E741
    else:
        labels = [int(l) - min_label for l in batch[label_col]]  # noqa: E741

    return {"input_values": all_iv, "attention_mask": all_am, "labels": labels}


def preprocess_splits(
    train_raw,
    val_raw,
    test_raw,
    label_col: str,
    all_labels,
    label2id: dict,
    feature_extractor,
):
    """Cast audio columns, featurise each split, and set torch format."""
    print("  Featurising splits ...")
    train_raw = train_raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    val_raw   = val_raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    test_raw  = test_raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    labels_are_strings = isinstance(all_labels[0], (str, np.str_))
    min_label = 0 if labels_are_strings else int(min(all_labels))

    fn = functools.partial(
        _preprocess_batch,
        feature_extractor=feature_extractor,
        label_col=label_col,
        label2id=label2id,
        labels_are_strings=labels_are_strings,
        min_label=min_label,
        max_length=MAX_LENGTH,
    )

    def _apply(ds, desc):
        d = ds.map(fn, batched=True, num_proc=1, remove_columns=ds.column_names, desc=desc)
        d.set_format("torch")
        return d

    train_ds = _apply(train_raw, "Train")
    val_ds   = _apply(val_raw,   "Val  ")
    test_ds  = _apply(test_raw,  "Test ")
    return train_ds, val_ds, test_ds
