"""Load the renumics/emodb (Berlin Database of Emotional Speech) dataset.

Returns the same list[dict] interface as data/ravdess.py and data/cameo.py so
that the MFCC pipeline (baselines/features.py -> baselines/train.py) can use
it directly.
"""

import io
import os
import sys

# ── Audio backend workaround (HPC clusters without FFmpeg / torchcodec) ───────
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
sys.modules["torchcodec"] = None  # type: ignore[assignment]

import datasets.config  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

datasets.config.TORCHCODEC_AVAILABLE = False
datasets.config.SOUNDFILE_AVAILABLE = True

from datasets import Audio, load_dataset  # noqa: E402


def _decode_soundfile(self, value, token_per_repo_id=None):
    if value is None:
        return value
    if isinstance(value, dict) and "array" in value:
        return value
    path = value.get("path")
    bytes_ = value.get("bytes")
    if bytes_:
        array, sr = sf.read(io.BytesIO(bytes_), dtype="float32", always_2d=False)
    elif path:
        array, sr = sf.read(path, dtype="float32", always_2d=False)
    else:
        return value
    target_sr = self.sampling_rate
    if target_sr and sr != target_sr:
        import librosa
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr, axis=0)
        sr = target_sr
    return {"path": path or "", "array": array.astype(np.float32), "sampling_rate": sr}


Audio.decode_example = _decode_soundfile
# ─────────────────────────────────────────────────────────────────────────────

EMODB_HF_NAME = "renumics/emodb"
SAMPLING_RATE = 16_000


def load() -> list[dict]:
    """Load emodb and return harmonized audio records.

    Returns:
        List of dicts with keys:
            "array" - float32 mono waveform at SAMPLING_RATE
            "sr"    - int sampling rate (always SAMPLING_RATE)
            "label" - emotion string
    """
    raw = load_dataset(EMODB_HF_NAME)
    base_split = list(raw.keys())[0]
    ds = raw[base_split]

    label_names = ds.features["emotion"].names
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    records = []
    for sample in ds:
        audio = sample["audio"]
        label = label_names[int(sample["emotion"])]
        arr = np.array(audio["array"], dtype=np.float32).squeeze()
        if arr.ndim > 1:
            arr = arr.mean(axis=1)  # (samples, channels) -> mono
        records.append({
            "array": arr,
            "sr": int(audio["sampling_rate"]),
            "label": label,
        })

    print(f"  emodb: loaded {len(records):,} samples")
    label_counts = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count:,}")

    return records
