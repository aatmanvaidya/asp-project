"""
Audio backend setup — import this module before any datasets/audio code.

Forces soundfile as the audio decoding backend and stubs torchcodec so the
pipeline works on HPC nodes where FFmpeg / torchcodec is unavailable.
Import order matters: os/sys must be patched before datasets submodules load.
"""
import io
import os
import sys

import numpy as np

os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"
sys.modules["torchcodec"] = None  # type: ignore[assignment]

import datasets.config  # noqa: E402
import soundfile as sf  # noqa: E402

datasets.config.TORCHCODEC_AVAILABLE = False
datasets.config.SOUNDFILE_AVAILABLE = True

from datasets import Audio  # noqa: E402


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
        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return {"path": path or "", "array": array.astype(np.float32), "sampling_rate": sr}


Audio.decode_example = _decode_soundfile
