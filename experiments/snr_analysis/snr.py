"""
Blind per-file SNR estimation.

Percentile-based frame-energy SNR, adapted from `sound_noise_ratio()` in
data/EmoDB_GitHub_EDA.ipynb (same 15th/50th percentile split, 20ms frames).
Kept in the power domain (variance per frame) rather than converting each
frame to dB first — equivalent, since dB is a monotonic transform of power
(percentile-then-log == log-then-percentile), just cheaper.
"""
import json
import os

import numpy as np
from tqdm import tqdm


def frame_snr_db(
    audio: np.ndarray,
    sample_rate: int,
    frame_duration: float = 0.02,
    noise_percentile: float = 15.0,
    speech_percentile: float = 50.0,
) -> float:
    """
    Split `audio` into non-overlapping `frame_duration`-second frames, and
    estimate SNR (dB) as the ratio between the speech_percentile-th and
    noise_percentile-th percentile of per-frame power (variance).
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    frame_size = int(frame_duration * sample_rate)
    n_frames = len(audio) // frame_size
    if frame_size <= 0 or n_frames == 0:
        return float("nan")

    frame_power = np.array([
        np.var(audio[i * frame_size:(i + 1) * frame_size]) for i in range(n_frames)
    ])

    noise_power = np.percentile(frame_power, noise_percentile)
    speech_power = np.percentile(frame_power, speech_percentile)
    if noise_power == 0:
        noise_power = 1e-10  # avoid divide-by-zero on true silence

    return float(10 * np.log10(speech_power / noise_power))


def compute_dataset_snr(hf_dataset: str, cache_path: str, force: bool = False) -> dict:
    """
    Per-file SNR (dB) for every sample in `hf_dataset`, keyed by audio
    filename (basename) — the same key used in test_predictions.csv's
    audio_file column. Cached to `cache_path` since it requires decoding
    every raw audio file.

    Caller must have already imported pipeline.audio_backend (patches the
    datasets audio decode backend) before calling this.
    """
    if os.path.exists(cache_path) and not force:
        with open(cache_path) as f:
            return json.load(f)

    from pipeline.data import load_dataset_hf

    raw, base_split, _label_col, _all_labels = load_dataset_hf(hf_dataset)
    base = raw[base_split]

    snr_by_file: dict[str, float] = {}
    skipped = 0
    for sample in tqdm(base, desc=f"SNR ({os.path.basename(cache_path)})"):
        audio = sample["audio"]
        filename = os.path.basename(audio.get("path") or "")
        if not filename:
            skipped += 1
            continue
        arr = np.asarray(audio["array"], dtype=np.float32)
        snr_by_file[filename] = frame_snr_db(arr, audio["sampling_rate"])

    if skipped:
        print(f"  WARNING: {skipped} samples had no audio path, skipped for SNR")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(snr_by_file, f, indent=2)
    print(f"  SNR computed for {len(snr_by_file):,} files -> {cache_path}")

    return snr_by_file
