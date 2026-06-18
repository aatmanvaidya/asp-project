import os
import sys

# ── FFmpeg / torchcodec workaround ────────────────────────────────────────────
# This HPC cluster has no FFmpeg, so torchcodec cannot be used. We must force
# the soundfile backend at every layer before anything else is imported.
#
# NOTE: The import order here is intentional and load-order-sensitive.
# os/sys must come first so we can stub torchcodec before datasets loads.
# sklearn/transformers imports are deferred below the datasets patch block
# because datasets.config must be patched before datasets submodules load.

# 1. Environment variable picked up by datasets at import time and by any
#    subprocess / worker that inherits the environment.
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

# 2. Stub out torchcodec so that `import torchcodec` raises ImportError
#    everywhere (parent process and forked workers).
sys.modules["torchcodec"] = None  # type: ignore[assignment]

import io

import datasets.config
import numpy as np
import soundfile as sf

# 3. Patch datasets config flags before the datasets package finishes loading.
datasets.config.TORCHCODEC_AVAILABLE = False
datasets.config.SOUNDFILE_AVAILABLE = True

from datasets import Audio, load_dataset

# 4. Monkey-patch Audio.decode_example so that even inside .map() worker
#    processes (which re-import datasets fresh) the torchcodec path is never
#    reached. We replace the method with one that always uses soundfile/librosa.


def _decode_example_soundfile(self, value, token_per_repo_id=None):
    """Decode audio using soundfile, bypassing torchcodec entirely."""
    if value is None:
        return value
    # value is already decoded (dict with 'array' and 'sampling_rate')
    if isinstance(value, dict) and "array" in value:
        return value
    # value is a raw bytes / path dict from the Arrow table
    path = value.get("path")
    bytes_ = value.get("bytes")
    if bytes_:
        array, sampling_rate = sf.read(
            io.BytesIO(bytes_), dtype="float32", always_2d=False
        )
    elif path:
        array, sampling_rate = sf.read(path, dtype="float32", always_2d=False)
    else:
        return value
    # Resample if needed
    target_sr = self.sampling_rate
    if target_sr is not None and sampling_rate != target_sr:
        import librosa

        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=target_sr)
        sampling_rate = target_sr
    return {
        "path": path or "",
        "array": array.astype(np.float32),
        "sampling_rate": sampling_rate,
    }


Audio.decode_example = _decode_example_soundfile
# ─────────────────────────────────────────────────────────────────────────────

DATASET_HF_NAME = "xbgoose/ravdess"
SAMPLING_RATE = 16_000

RAVDESS_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def load() -> list[dict]:
    raw = load_dataset(DATASET_HF_NAME)
    base_split = list(raw.keys())[0]
    ds = raw[base_split]

    features = ds.features
    label_col = next(
        (c for c in ["label", "emotion", "labels", "target"] if c in features),
        None,
    )
    assert label_col is not None, "Could not detect label column."

    all_labels = ds.with_format("numpy")[label_col]
    labels_are_strings = isinstance(all_labels[0], (str, np.str_))

    if hasattr(features.get(label_col), "names"):
        label_names = features[label_col].names
        id2label = {i: name for i, name in enumerate(label_names)}
        min_label = 0
        labels_are_strings = False
    elif labels_are_strings:
        label_names = sorted(set(str(l) for l in all_labels))  # noqa: E741
        id2label = None
        min_label = 0
    else:
        min_label = int(min(all_labels))
        n_labels = int(max(all_labels)) - min_label + 1
        label_names = RAVDESS_NAMES[:n_labels]
        id2label = {i: name for i, name in enumerate(label_names)}

    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    records = []
    for sample in ds:
        audio = sample["audio"]
        label_raw = sample[label_col]
        if labels_are_strings:
            label_str = str(label_raw)
        else:
            label_str = id2label[int(label_raw) - min_label]
        arr = np.array(audio["array"], dtype=np.float32).squeeze()
        if arr.ndim > 1:
            arr = arr.mean(axis=1)  # (samples, channels) → mono
        records.append({
            "array": arr,
            "sr": int(audio["sampling_rate"]),
            "label": label_str,
        })
    return records
