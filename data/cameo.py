"""Load and harmonize the amu-cai/CAMEO multi-lingual speech emotion collection.

Returns the same list[dict] interface as data/ravdess.py so that the MFCC
pipeline (baselines/features.py → baselines/train.py) can use it directly.

Label harmonization
-------------------
CAMEO sub-datasets use slightly different emotion names.  We normalise all of
them to a shared canonical set and drop samples whose labels cannot be mapped.

Canonical labels: anger, disgust, fear, happiness, neutral, sadness, surprise
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

CAMEO_HF_NAME = "amu-cai/CAMEO"
SAMPLING_RATE = 16_000

CANONICAL_LABELS = frozenset(
    ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]
)

# Map dataset-specific variants to canonical form; None means drop.
_LABEL_ALIASES: dict[str, str | None] = {
    "angry": "anger",
    "happy": "happiness",
    "joy": "happiness",
    "sad": "sadness",
    "fearful": "fear",
    "scared": "fear",
    "surprised": "surprise",
    # drop non-standard emotions
    "calm": None,
    "contempt": None,
    "boredom": None,
    "bored": None,
    "enthusiasm": None,
    "excited": None,
    "poker": None,
    "amused": None,
    "sleepy": None,
}


def _normalize(raw_label: str) -> str | None:
    """Return the canonical label or None if the sample should be dropped."""
    label = raw_label.strip().lower()
    if label in CANONICAL_LABELS:
        return label
    return _LABEL_ALIASES.get(label, None)  # None for unknown labels → drop


def load() -> list[dict]:
    """Load all CAMEO sub-datasets and return harmonized audio records.

    Returns:
        List of dicts with keys:
            "array"  – float32 mono waveform at SAMPLING_RATE
            "sr"     – int sampling rate (always SAMPLING_RATE)
            "label"  – canonical emotion string
            "source" – name of the originating CAMEO sub-dataset
    """
    raw = load_dataset(CAMEO_HF_NAME)

    records: list[dict] = []
    skipped_splits: list[str] = []
    dropped = 0

    for split_name, dataset in raw.items():
        if "emotion" not in dataset.column_names or "audio" not in dataset.column_names:
            skipped_splits.append(split_name)
            continue

        dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

        for sample in dataset:
            label = _normalize(str(sample.get("emotion", "")))
            if label is None:
                dropped += 1
                continue

            audio = sample["audio"]
            arr = np.array(audio["array"], dtype=np.float32).squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=1)

            records.append({
                "array": arr,
                "sr": int(audio["sampling_rate"]),
                "label": label,
                "source": split_name,
            })

    if skipped_splits:
        print(f"  CAMEO: skipped splits (missing audio/emotion columns): {skipped_splits}")
    print(f"  CAMEO: loaded {len(records):,} samples, dropped {dropped:,} with non-canonical labels")

    label_counts = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count:,}")

    return records
